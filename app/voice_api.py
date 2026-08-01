"""
Voice REST endpoints (legacy fallback layer).

Primary interface:  WebSocket at /ws/voice  (see voice_ws.py)
Fallback interface: POST /voice-stream, /auth/voice-stream  (binary framed stream, MP3 audio)
"""

from __future__ import annotations

import json
import logging
import re
import struct
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.student_messages import EMPTY_VOICE_MESSAGE

from app.services.edge_tts_service import (
    FALLBACK_VOICE,
    PRIMARY_VOICE,
    iter_edge_tts_mp3,
    voice_for_gender,
)
from app.services.tts_sanitize import sanitize_chunk_for_tts
from app.services.voice_chunking import extract_voice_chunks
from app.services.voice_http_tts import http_tts_prefetcher
from app.services.voice_stt_postprocess import postprocess_voice_transcript

logger = logging.getLogger(__name__)

router = APIRouter(prefix="")

_FRAME_TEXT = 1
_FRAME_AUDIO = 2  # MP3 chunk (audio/mpeg)
_FRAME_DONE = 3
_FRAME_IMAGES = 4
_FRAME_TUTOR_HINT = 5
_FRAME_TUTOR_STATE = 6
_FRAME_MATH_LESSON = 7
_FRAME_SCIENCE_EXPERIMENT = 8


def _frame(ftype: int, data: bytes) -> bytes:
    return struct.pack("<BI", ftype, len(data)) + data


def _split_sentences(buf: str) -> tuple[list[str], str]:
    parts = re.split(r"(?<=[.?!])\s+", buf)
    if len(parts) > 1:
        return [s.strip() for s in parts[:-1] if s.strip()], parts[-1]
    return [], buf


async def _stream_mp3_frames(text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
    async for chunk in iter_edge_tts_mp3(text, voice=voice):
        yield _frame(_FRAME_AUDIO, chunk)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class ChatVoiceRequest(BaseModel):
    message: str


async def _chat_voice_mp3_stream(text: str) -> AsyncIterator[bytes]:
    """Stream MP3 bytes as Edge-TTS generates them (no full-buffer join)."""
    import time

    from app.services.tts_sanitize import sanitize_for_tts

    spoken = sanitize_for_tts(text)
    if not spoken:
        return

    t0 = time.monotonic()
    first = True
    async for chunk in iter_edge_tts_mp3(spoken):
        if first:
            logger.info(
                "[chat-voice] first MP3 chunk @ %.0fms (chars=%d)",
                (time.monotonic() - t0) * 1000,
                len(text),
            )
            first = False
        yield chunk
    logger.info("[chat-voice] stream complete @ %.0fms", (time.monotonic() - t0) * 1000)


@router.post("/chat-voice")
async def chat_voice(req: ChatVoiceRequest):
    """
    Stream assistant text to MP3 via Edge TTS (read-aloud in text chat).

    Returns chunked audio/mpeg; clients should play progressively, not wait for full body.
    """
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail=EMPTY_VOICE_MESSAGE)

    return StreamingResponse(
        _chat_voice_mp3_stream(text),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'inline; filename="voice.mp3"',
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


class ConversationTurn(BaseModel):
    role: str
    content: str


class ChapterVoiceRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    board: str = ""
    class_level: str = ""
    subject_name: str = ""
    chapter_ids: list[str] | None = None
    chapter: str = ""
    chapter_names: list[str] | None = None
    student_name: str = ""
    tutor_state: str = "TEACHING"
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    voice_gender: str = ""
    tts_voice: str = ""


def _normalize_history(turns: list[ConversationTurn] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in turns or []:
        role = (turn.role or "").lower()
        content = (turn.content or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out[-14:]


async def _stream_answer_frames(
    answer: str,
    imgs: list | None = None,
    *,
    voice: str | None = None,
) -> AsyncIterator[bytes]:
    if imgs is not None:
        yield _frame(_FRAME_IMAGES, json.dumps(imgs).encode("utf-8"))
    for word in answer.split():
        yield _frame(_FRAME_TEXT, (word + " ").encode())
    for part in re.split(r"(?<=[.?!])\s+", answer):
        if not part.strip():
            continue
        async for framed in _stream_mp3_frames(part.strip(), voice=voice):
            yield framed
    yield _frame(_FRAME_DONE, b"")


async def _chapter_voice_stream_generate(req: ChapterVoiceRequest) -> AsyncIterator[bytes]:
    from app.services.chapter_scope import resolve_chapter_scope_with_retrieval
    from app.services.chat_service import _voice_should_use_text_format, chapter_aware_qa_stream
    from app.services.voice_tutor import (
        TutorState,
        evaluate_student_response,
        next_tutor_state,
    )

    message = postprocess_voice_transcript(req.message, subject_name=req.subject_name)
    if len(message.strip()) < 2:
        return

    tts_voice = voice_for_gender(req.voice_gender, req.tts_voice or None)

    history = _normalize_history(req.conversation_history)
    last_assistant = ""
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            last_assistant = turn.get("content") or ""
            break

    try:
        current_state = TutorState(req.tutor_state)
    except ValueError:
        current_state = TutorState.TEACHING

    scores = evaluate_student_response(
        message,
        last_assistant=last_assistant,
        tutor_state=current_state,
    )
    hint = scores.to_student_hint(message)
    if hint:
        yield _frame(_FRAME_TUTOR_HINT, hint.encode("utf-8"))

    understanding_payload = {
        "understanding": scores.understanding,
        "confidence": scores.confidence,
        "confusion": scores.confusion,
        "is_affirmation": scores.is_affirmation,
        "wants_expansion": scores.wants_expansion,
        "wants_quiz": scores.wants_quiz,
    }
    use_text_format = _voice_should_use_text_format(
        req.subject_name,
        voice_mode=True,
        understanding_scores=understanding_payload,
        query=message,
    )

    collection = f"{req.board}_{req.class_level}_{req.subject_name}".replace(" ", "_")

    if req.board and req.subject_name and req.chapter_ids:
        scope_msg = resolve_chapter_scope_with_retrieval(
            message,
            collection_name=collection,
            chapter_ids=req.chapter_ids,
            chapter_names=req.chapter_names,
            board=req.board,
            class_level=req.class_level,
            subject_name=req.subject_name,
            conversation_history=history,
        )
        if scope_msg:
            async for framed in _stream_answer_frames(scope_msg, [], voice=tts_voice):
                yield framed
            return

    sentence_buf = ""
    full_answer_parts: list[str] = []
    speech_units_emitted = 0
    side_frames: list[bytes] = []
    tts_prefetcher = http_tts_prefetcher()

    async def _yield_speech_units(units: list[str]) -> AsyncIterator[bytes]:
        nonlocal speech_units_emitted
        batch = [u for u in units if u.strip()]
        if not batch:
            return
        speech_units_emitted += len(batch)
        async for framed in tts_prefetcher.stream_units(batch, voice=tts_voice):
            yield framed

    async def emit_imgs(imgs: list[dict]) -> None:
        side_frames.append(_frame(_FRAME_IMAGES, json.dumps(imgs).encode("utf-8")))

    async def emit_lesson(lesson: dict | None, clean_answer: str) -> None:
        if not lesson:
            return
        payload = json.dumps(
            {"lesson": lesson, "clean_answer": clean_answer},
            ensure_ascii=False,
        ).encode("utf-8")
        side_frames.append(_frame(_FRAME_MATH_LESSON, payload))

    async def emit_experiment(experiment: dict | None, clean_answer: str) -> None:
        if not experiment:
            return
        payload = json.dumps(
            {"experiment": experiment, "clean_answer": clean_answer},
            ensure_ascii=False,
        ).encode("utf-8")
        side_frames.append(_frame(_FRAME_SCIENCE_EXPERIMENT, payload))

    async for token in chapter_aware_qa_stream(
        message,
        collection_name=collection,
        chapter_ids=req.chapter_ids,
        class_level=req.class_level,
        board=req.board,
        subject_name=req.subject_name,
        chapter=req.chapter,
        chapter_names=req.chapter_names,
        emit_related_images=emit_imgs,
        emit_math_lesson=emit_lesson,
        emit_science_experiment=emit_experiment,
        voice_mode=True,
        tutor_state=current_state.value,
        understanding_scores=understanding_payload,
        conversation_history=history,
        student_name=req.student_name,
    ):
        while side_frames:
            yield side_frames.pop(0)
        yield _frame(_FRAME_TEXT, token.encode("utf-8"))
        full_answer_parts.append(token)
        sentence_buf += token

        chunks, sentence_buf = extract_voice_chunks(
            sentence_buf, chunks_emitted=speech_units_emitted
        )
        if chunks:
            async for framed in _yield_speech_units(chunks):
                yield framed

    while side_frames:
        yield side_frames.pop(0)

    if sentence_buf.strip():
        cleaned = sanitize_chunk_for_tts(sentence_buf.strip())
        if cleaned:
            async for framed in _yield_speech_units([cleaned]):
                yield framed

    full_answer = "".join(full_answer_parts)
    next_state = next_tutor_state(
        current=current_state,
        scores=scores,
        assistant_reply=full_answer,
    )
    yield _frame(_FRAME_TUTOR_STATE, next_state.value.encode("utf-8"))
    yield _frame(_FRAME_DONE, b"")


@router.post("/auth/voice-stream")
async def chapter_voice_stream(req: ChapterVoiceRequest):
    """
    Chapter-aware voice stream (POST fallback when WebSocket is unavailable).

    Binary frames:
        FRAME_TEXT       (1) – UTF-8 text token
        FRAME_AUDIO      (2) – MP3 chunk
        FRAME_DONE       (3) – end of stream
        FRAME_IMAGES     (4) – JSON related images
        FRAME_TUTOR_HINT (5) – UTF-8 student-facing hint (e.g. I heard: "...")
        FRAME_TUTOR_STATE (6) – UTF-8 tutor state for next HTTP turn
        FRAME_MATH_LESSON (7) – JSON {lesson, clean_answer}
        FRAME_SCIENCE_EXPERIMENT (8) – JSON {experiment, clean_answer}
    """
    return StreamingResponse(
        _chapter_voice_stream_generate(req),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/voice-stream")
async def general_voice_stream(req: ChapterVoiceRequest):
    """Unauthenticated/general voice stream fallback (same framed protocol)."""
    return StreamingResponse(
        _chapter_voice_stream_generate(req),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/voice/tts-info")
def tts_info() -> dict[str, str]:
    from app.config import VOICE_TTS_PITCH, VOICE_TTS_RATE

    return {
        "provider": "edge-tts",
        "primary_voice": PRIMARY_VOICE,
        "fallback_voice": FALLBACK_VOICE,
        "audio_format": "audio/mpeg",
        "rate": VOICE_TTS_RATE,
        "pitch": VOICE_TTS_PITCH,
    }


@router.get("/voice/stt-info")
def stt_info() -> dict[str, str | bool]:
    from app.services.voice_whisper_stt import whisper_status

    return whisper_status()


@router.get("/voice/protection-info")
def protection_info() -> dict:
    """Central voice-protection flags (VAD / echo / speaker / noise / wake)."""
    from app.config import (
        ECHO_SIMILARITY_THRESHOLD,
        INTERRUPT_SCORE_THRESHOLD,
        INTERRUPT_WEIGHT_INTENT,
        INTERRUPT_WEIGHT_SPEAKER,
        INTERRUPT_WEIGHT_VAD,
        MIN_SPEECH_MS,
        NOISE_SUPPRESSION_ENABLED,
        NOISE_SUPPRESSION_PROVIDER,
        POST_PLAYBACK_STT_DELAY_MS,
        SPEAKER_SIMILARITY_THRESHOLD,
        SPEAKER_VERIFICATION_ENABLED,
        VAD_ENABLED,
        VAD_THRESHOLD,
        VOICE_PROTECTION_ENABLED,
        VOICE_SESSION_PROFILE,
        VOICE_SESSION_PROFILE_MIN_MS,
        VOICE_SESSION_REDIS,
        VOICE_TTS_PREFETCH_DEPTH,
        WAKE_WORD_ENABLED,
        WAKE_WORDS,
    )
    from app.services.voice_noise_suppress import noise_status
    from app.services.voice_protection_preload import preload_status
    from app.services.voice_speaker import speaker_status
    from app.services.voice_vad import vad_status

    return {
        "VOICE_PROTECTION_ENABLED": VOICE_PROTECTION_ENABLED,
        "VAD_ENABLED": VAD_ENABLED,
        "VAD_THRESHOLD": VAD_THRESHOLD,
        "MIN_SPEECH_MS": MIN_SPEECH_MS,
        "POST_PLAYBACK_STT_DELAY_MS": POST_PLAYBACK_STT_DELAY_MS,
        "ECHO_SIMILARITY_THRESHOLD": ECHO_SIMILARITY_THRESHOLD,
        "SPEAKER_VERIFICATION_ENABLED": SPEAKER_VERIFICATION_ENABLED,
        "SPEAKER_SIMILARITY_THRESHOLD": SPEAKER_SIMILARITY_THRESHOLD,
        "NOISE_SUPPRESSION_ENABLED": NOISE_SUPPRESSION_ENABLED,
        "NOISE_SUPPRESSION_PROVIDER": NOISE_SUPPRESSION_PROVIDER,
        "INTERRUPT_SCORE_THRESHOLD": INTERRUPT_SCORE_THRESHOLD,
        "INTERRUPT_WEIGHTS": {
            "vad": INTERRUPT_WEIGHT_VAD,
            "speaker": INTERRUPT_WEIGHT_SPEAKER,
            "intent": INTERRUPT_WEIGHT_INTENT,
        },
        "VOICE_TTS_PREFETCH_DEPTH": VOICE_TTS_PREFETCH_DEPTH,
        "WAKE_WORD_ENABLED": WAKE_WORD_ENABLED,
        "WAKE_WORDS": WAKE_WORDS,
        "VOICE_SESSION_PROFILE": VOICE_SESSION_PROFILE,
        "VOICE_SESSION_PROFILE_MIN_MS": VOICE_SESSION_PROFILE_MIN_MS,
        "VOICE_SESSION_REDIS": VOICE_SESSION_REDIS,
        "vad": vad_status(),
        "speaker": speaker_status(),
        "noise": noise_status(),
        "models_loaded": preload_status(),
    }


@router.get("/voice/protection-metrics")
def protection_metrics() -> dict:
    from app.services.voice_protection_metrics import snapshot

    return snapshot()


@router.post("/voice/barge-check")
async def voice_barge_check(
    audio: UploadFile = File(...),
    transcript: str = Form(default=""),
    recent_ai_speech: str = Form(default=""),
    student_key: str = Form(default=""),
    voice_session_id: str = Form(default=""),
):
    """
    Gate a barge-in candidate: VAD → echo → speaker → intent.
    Existing interrupt WS path unchanged — client calls this before interrupting.
    """
    from app.services.voice_interrupt_pipeline import evaluate_barge_in

    data = await audio.read()
    if len(data) < 64:
        return {"allow_interrupt": False, "reason": "audio_too_short"}
    result = evaluate_barge_in(
        data,
        transcript=transcript,
        recent_ai_speech=recent_ai_speech,
        student_key=student_key,
        voice_session_id=voice_session_id.strip(),
    )
    # Don't ship processed PCM back over JSON
    result.pop("processed_audio", None)
    return result


@router.post("/auth/voice-enroll")
async def auth_voice_enroll(
    audio: UploadFile = File(...),
    student_key: str = Form(default=""),
):
    """Enroll speaker embedding: \"Hello, I am ready to learn.\""""
    from app.services.voice_speaker import ENROLLMENT_PROMPT, enroll_speaker

    data = await audio.read()
    if len(data) < 256:
        raise HTTPException(status_code=400, detail="Audio too short for enrollment.")
    key = (student_key or "").strip() or "anonymous"
    result = enroll_speaker(key, data)
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error") or "Enrollment failed")
    result["prompt"] = ENROLLMENT_PROMPT
    return result


@router.post("/auth/voice-verify")
async def auth_voice_verify(
    audio: UploadFile = File(...),
    student_key: str = Form(default=""),
):
    from app.services.voice_speaker import verify_speaker

    data = await audio.read()
    key = (student_key or "").strip() or "anonymous"
    return verify_speaker(key, data)


async def _handle_voice_transcribe(
    audio: UploadFile,
    *,
    subject_name: str = "",
    language: str = "en",
    reject_if_similar_to: str = "",
    voice_session_id: str = "",
) -> dict[str, str | float | bool]:
    from app.services.voice_session_profile import bootstrap_session_voice
    from app.services.voice_whisper_stt import transcribe_audio_bytes, whisper_available

    if not whisper_available():
        raise HTTPException(
            status_code=503,
            detail="Server Whisper STT is not available. Install faster-whisper or use browser speech.",
        )
    data = await audio.read()
    if len(data) < 256:
        raise HTTPException(status_code=400, detail="Audio too short to transcribe.")

    sid = (voice_session_id or "").strip()
    try:
        result = await transcribe_audio_bytes(
            data,
            language=language or "en",
            subject_name=subject_name,
            reject_if_similar_to=reject_if_similar_to,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if "Invalid data" in str(exc) or exc.__class__.__name__ == "InvalidDataError":
            raise HTTPException(status_code=422, detail="Could not read the audio clip.") from exc
        raise
    transcript = str(result.get("transcript") or "").strip()
    if result.get("rejected"):
        return {
            "transcript": "",
            "provider": "faster-whisper",
            "confidence": result.get("confidence", 0.0),
            "rejected": True,
        }
    if not transcript:
        return {
            "transcript": "",
            "provider": "faster-whisper",
            "confidence": result.get("confidence", 0.0),
            "rejected": False,
        }

    if sid:
        bootstrap_session_voice(sid, data)

    return {
        "transcript": transcript,
        "provider": "faster-whisper",
        "confidence": result.get("confidence", 0.0),
        "rejected": False,
    }


@router.post("/auth/voice-transcribe")
async def auth_voice_transcribe(
    audio: UploadFile = File(...),
    subject_name: str = Form(default=""),
    language: str = Form(default="en"),
    reject_if_similar_to: str = Form(default=""),
    voice_session_id: str = Form(default=""),
):
    """Transcribe a short microphone clip (WebM/Opus) with Whisper — better math accuracy."""
    return await _handle_voice_transcribe(
        audio,
        subject_name=subject_name,
        language=language,
        reject_if_similar_to=reject_if_similar_to,
        voice_session_id=voice_session_id,
    )


@router.post("/voice-transcribe")
async def general_voice_transcribe(
    audio: UploadFile = File(...),
    subject_name: str = Form(default=""),
    language: str = Form(default="en"),
    reject_if_similar_to: str = Form(default=""),
    voice_session_id: str = Form(default=""),
):
    """Unauthenticated Whisper transcribe (same as /auth/voice-transcribe)."""
    return await _handle_voice_transcribe(
        audio,
        subject_name=subject_name,
        language=language,
        reject_if_similar_to=reject_if_similar_to,
        voice_session_id=voice_session_id,
    )


@router.post("/auth/voice-session-end")
async def auth_voice_session_end(voice_session_id: str = Form(default="")):
    """Delete ephemeral session voice profile when the student leaves AI Voice."""
    from app.services.voice_session_profile import clear_session_voice

    sid = (voice_session_id or "").strip()
    if sid:
        clear_session_voice(sid)
    return {"ok": True}


@router.post("/voice-session-end")
async def voice_session_end(voice_session_id: str = Form(default="")):
    from app.services.voice_session_profile import clear_session_voice

    sid = (voice_session_id or "").strip()
    if sid:
        clear_session_voice(sid)
    return {"ok": True}

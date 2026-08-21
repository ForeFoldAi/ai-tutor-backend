"""
Real-time AI Voice Tutor via WebSocket.

Concurrent pipeline:
  LLM tokens → speech-unit chunking → asyncio.Queue → TTS orchestrator (prefetch) → MP3 frames
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.config import VOICE_INTERRUPT_CANCEL_SEC
from app.core.student_messages import VOICE_ANSWER_FAILED, VOICE_SERVER_ERROR

from app.services.edge_tts_service import stream_edge_tts, voice_for_gender
from app.services.voice_stt_postprocess import postprocess_voice_transcript
from app.services.voice_chunking import VoicePipelineTiming, extract_voice_chunks
from app.services.voice_streaming import (
    SpeechTokenBuffer,
    create_speech_unit_queue,
    enqueue_speech_unit,
    flush_units_from_buffer,
    idle_flush_loop,
)
from app.services.voice_tts_orchestrator import clear_speech_queue, tts_worker

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# Queue sentinel: end TTS worker for this turn (shared with voice_tts_orchestrator)
_TTS_STOP = None


async def _emit_stream_metrics(
    ws: WebSocket,
    timing: VoicePipelineTiming,
    phase: str,
    **extra: str,
) -> None:
    from app.services.voice_performance import export_voice_turn_metrics

    export_voice_turn_metrics(timing, phase=phase, **extra)
    await _send(
        ws,
        {"type": "stream_metrics", "metrics": timing.summary(), "phase": phase},
    )


async def _run_llm_producer(
    ws: WebSocket,
    token_iter: AsyncIterator[str],
    sentence_queue: asyncio.Queue[str | None],
    stop: asyncio.Event,
    timing: VoicePipelineTiming,
    full_tokens: list[str],
) -> None:
    """Producer: stream LLM tokens → speech units (Phase 2 bounded queue + metrics)."""
    buffer = SpeechTokenBuffer()

    async def emit_metrics(phase: str) -> None:
        await _emit_stream_metrics(ws, timing, phase)

    idle_task = asyncio.create_task(
        idle_flush_loop(buffer, sentence_queue, stop, timing, emit_metrics=emit_metrics),
        name="voice-idle-flush",
    )
    first_token = True

    try:
        async for token in token_iter:
            if stop.is_set():
                return

            if first_token:
                timing.mark_llm_first_token()
                first_token = False

            full_tokens.append(token)
            await _send(ws, {"type": "ai_text_token", "token": token})

            await buffer.append_token(token)
            timing.record_tokens_streamed(buffer.tokens_streamed)
            await flush_units_from_buffer(
                buffer, sentence_queue, timing, emit_metrics=emit_metrics
            )

        async with buffer.lock:
            remainder = buffer.buf.strip()
            buffer.buf = ""

        if remainder and not stop.is_set():
            await enqueue_speech_unit(
                sentence_queue,
                timing,
                remainder,
                buffer,
                emit_metrics=emit_metrics,
            )

    finally:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass


class _Session:
    __slots__ = (
        "board",
        "class_level",
        "subject_name",
        "chapter_ids",
        "chapter",
        "chapter_names",
        "history",
        "memory",
        "student_name",
        "student_key",
        "tutor_state",
        "last_topic",
        "learner_profile",
        "_learner_load_task",
        "tts_voice",
        "voice_session_id",
    )

    def __init__(self) -> None:
        from app.services.conversation_memory import ConversationMemory

        self.board = ""
        self.class_level = ""
        self.subject_name = ""
        self.chapter_ids: list[str] | None = None
        self.chapter = ""
        self.chapter_names: list[str] = []
        self.history: list[dict] = []
        self.memory = ConversationMemory()
        self.student_name = ""
        self.student_key = ""
        self.tutor_state = "LISTENING"
        self.last_topic = ""
        self.learner_profile = None
        self._learner_load_task: asyncio.Task | None = None
        self.tts_voice: str | None = None
        self.voice_session_id: str = ""

    def configure(self, msg: dict) -> None:
        self.board = msg.get("board", self.board)
        self.class_level = msg.get("class_level", self.class_level)
        self.subject_name = msg.get("subject_name", self.subject_name)
        self.student_name = msg.get("student_name", self.student_name) or self.student_name
        self.chapter_ids = msg.get("chapter_ids", self.chapter_ids)
        self.chapter = msg.get("chapter", self.chapter)
        raw_names = msg.get("chapter_names")
        if isinstance(raw_names, list):
            self.chapter_names = [str(x) for x in raw_names if str(x).strip()]
        elif isinstance(raw_names, str) and raw_names.strip():
            self.chapter_names = [raw_names.strip()]
        picked = voice_for_gender(msg.get("voice_gender"), msg.get("tts_voice"))
        if picked is not None:
            self.tts_voice = picked
        sid = str(msg.get("voice_session_id") or "").strip()
        if sid:
            self.voice_session_id = sid

    def remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 20:
            self.history = self.history[-20:]

    @property
    def has_context(self) -> bool:
        return bool(self.board and self.subject_name)

    @property
    def collection(self) -> str:
        return f"{self.board}_{self.class_level}_{self.subject_name}".replace(" ", "_")


async def _send(ws: WebSocket, payload: dict) -> None:
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except Exception:
        pass


async def _general_answer_stream(
    question: str,
    session: _Session,
    *,
    understanding_scores: dict | None = None,
    learner_snapshot: dict | None = None,
) -> AsyncIterator[str]:
    """General (no-chapter) voice answers — same format as text chat."""
    from app.services.chat_service import _build_chat_messages, _stream_mistral_async

    del understanding_scores, learner_snapshot

    messages = _build_chat_messages(
        question,
        "(No chapter excerpt — use accurate general knowledge briefly.)",
        class_level=session.class_level,
        board=session.board,
        subject_name=session.subject_name,
        chapter=session.chapter,
        student_name=session.student_name,
        conversation_history=session.history,
    )
    async for token in _stream_mistral_async(messages, feature="voice"):
        yield token


async def _play_session_greeting(ws: WebSocket, session: _Session) -> None:
    """Welcome message with TTS when the student opens AI Voice from Start Learning."""
    from app.services.chat_service import build_session_greeting

    text = build_session_greeting(
        student_name=session.student_name,
        subject_name=session.subject_name,
        chapter=session.chapter,
        chapter_names=session.chapter_names,
    )
    stop = asyncio.Event()
    timing = VoicePipelineTiming(turn_id="greeting")
    await _send(ws, {"type": "greeting_start", "text": text})
    try:
        await _send(ws, {"type": "speaking"})
        await stream_edge_tts(text, ws, stop, voice=session.tts_voice, timing=timing)
    finally:
        if not stop.is_set():
            await _send(ws, {"type": "done"})
            await _send(ws, {"type": "listening"})
    session.remember("assistant", text)


async def _pipeline_cached_answer(
    ws: WebSocket,
    answer: str,
    stop: asyncio.Event,
    timing: VoicePipelineTiming,
    *,
    tts_voice: str | None = None,
) -> None:
    speaking = False

    async def on_speaking() -> None:
        nonlocal speaking
        if not speaking:
            await _send(ws, {"type": "speaking"})
            speaking = True

    sentence_queue = create_speech_unit_queue()
    tts_task = asyncio.create_task(
        tts_worker(ws, sentence_queue, stop, timing, on_speaking, voice=tts_voice),
        name="voice-tts-worker",
    )

    for word in answer.split():
        if stop.is_set():
            break
        await _send(ws, {"type": "ai_text_token", "token": word + " "})

    buffer = SpeechTokenBuffer()
    buf = answer
    while buf.strip() and not stop.is_set():
        chunks, buf = extract_voice_chunks(buf, chunks_emitted=buffer.chunks_emitted)
        for s in chunks:
            await enqueue_speech_unit(sentence_queue, timing, s, buffer)
        if not chunks:
            break
    if buf.strip() and not stop.is_set():
        await enqueue_speech_unit(sentence_queue, timing, buf.strip(), buffer)

    await sentence_queue.put(_TTS_STOP)
    await tts_task


async def _stream_answer(
    ws: WebSocket,
    question: str,
    session: _Session,
    stop: asyncio.Event,
) -> None:
    from app.services.chat_service import chapter_aware_qa_stream
    from app.services.learner_profile import save_learner_profile
    from app.services.voice_tutor import (
        TutorState,
        evaluate_student_response,
        next_tutor_state,
    )

    turn_id = uuid.uuid4().hex[:8]
    timing = VoicePipelineTiming(turn_id=turn_id)
    logger.info("[voice %s] Turn started state=%s", turn_id, session.tutor_state)

    last_assistant = ""
    for turn in reversed(session.history):
        if (turn.get("role") or "").lower() == "assistant":
            last_assistant = (turn.get("content") or "").strip()
            break

    try:
        current_state = TutorState(session.tutor_state)
    except ValueError:
        current_state = TutorState.LISTENING

    scores = evaluate_student_response(
        question,
        last_assistant=last_assistant,
        tutor_state=current_state,
    )

    if session.student_key:
        from app.services.voice_performance import resolve_learner_profile

        await resolve_learner_profile(session)
        if session.learner_profile is not None:
            topic = session.last_topic or question
            session.learner_profile.apply_understanding(topic, scores)
            # ponytail: fire-and-forget save — don't block first token
            asyncio.create_task(
                save_learner_profile(session.learner_profile),
                name="voice-learner-save",
            )

    session.last_topic = question
    understanding_payload = {
        "understanding": scores.understanding,
        "confidence": scores.confidence,
        "confusion": scores.confusion,
        "is_affirmation": scores.is_affirmation,
        "wants_expansion": scores.wants_expansion,
        "wants_quiz": scores.wants_quiz,
    }
    learner_snapshot = (
        session.learner_profile.snapshot().__dict__
        if session.learner_profile
        else None
    )

    if stop.is_set():
        return

    student_hint = scores.to_student_hint(question)
    if student_hint:
        await _send(ws, {"type": "tutor_hint", "hint": student_hint})

    await _send(ws, {"type": "thinking"})

    full_tokens: list[str] = []
    clean_answer_holder: list[str] = []
    speaking = False

    async def on_speaking() -> None:
        nonlocal speaking
        if not speaking:
            await _send(ws, {"type": "speaking"})
            speaking = True

    sentence_queue = create_speech_unit_queue()
    tts_task = asyncio.create_task(
        tts_worker(ws, sentence_queue, stop, timing, on_speaking, voice=session.tts_voice),
        name="voice-tts-worker",
    )

    async def _emit_imgs(imgs: list[dict]) -> None:
        if stop.is_set():
            return
        await _send(ws, {"type": "related_images", "images": imgs})

    async def _emit_lesson(lesson: dict | None, clean_answer: str) -> None:
        if stop.is_set() or not lesson:
            return
        if clean_answer:
            clean_answer_holder.clear()
            clean_answer_holder.append(clean_answer)
        await _send(
            ws,
            {
                "type": "math_lesson",
                "lesson": lesson,
                "clean_answer": clean_answer,
            },
        )

    async def _emit_experiment(experiment: dict | None, clean_answer: str) -> None:
        if stop.is_set() or not experiment:
            return
        if clean_answer:
            clean_answer_holder.clear()
            clean_answer_holder.append(clean_answer)
        await _send(
            ws,
            {
                "type": "science_experiment",
                "experiment": experiment,
                "clean_answer": clean_answer,
            },
        )

    try:
        if session.has_context:
            token_iter = chapter_aware_qa_stream(
                question,
                collection_name=session.collection,
                chapter_ids=session.chapter_ids,
                class_level=session.class_level,
                board=session.board,
                subject_name=session.subject_name,
                chapter=session.chapter,
                chapter_names=session.chapter_names,
                emit_related_images=_emit_imgs,
                emit_math_lesson=_emit_lesson,
                emit_science_experiment=_emit_experiment,
                conversation_history=session.history,
                conversation_memory=session.memory.to_dict(),
                student_name=session.student_name,
                student_key=session.student_key,
                voice_mode=True,
                tutor_state=session.tutor_state,
                understanding_scores=understanding_payload,
                learner_snapshot=learner_snapshot,
                pipeline_timing=timing,
            )
        else:
            token_iter = _general_answer_stream(
                question,
                session,
                understanding_scores=understanding_payload,
                learner_snapshot=learner_snapshot,
            )

        await _run_llm_producer(ws, token_iter, sentence_queue, stop, timing, full_tokens)

    except asyncio.CancelledError:
        await clear_speech_queue(sentence_queue)
        raise
    except Exception as exc:
        logger.error("Voice generation error: %s", exc, exc_info=True)
        if not stop.is_set():
            await _send(ws, {"type": "error", "message": VOICE_ANSWER_FAILED})
        return
    finally:
        await sentence_queue.put(_TTS_STOP)
        try:
            await asyncio.wait_for(tts_task, timeout=120.0)
        except asyncio.TimeoutError:
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

    if stop.is_set():
        logger.info("[voice %s] Interrupted — %s", turn_id, timing.summary())
        return

    await _emit_stream_metrics(ws, timing, "turn_end")
    await _send(ws, {"type": "done"})
    logger.info("[voice %s] Timing %s", turn_id, timing.summary())

    full_answer = clean_answer_holder[0] if clean_answer_holder else "".join(full_tokens)
    from app.services.conversation_context import resolve_conversation_context
    from app.services.conversation_memory import remember_turn

    turn_conv = resolve_conversation_context(
        question,
        conversation_history=session.history,
        chapter=session.chapter,
        memory=session.memory,
    )
    session.history, session.memory = remember_turn(
        session.history,
        session.memory,
        user_query=question,
        assistant_response=full_answer,
        resolved_topic=turn_conv.resolved_topic,
        followup_type=turn_conv.followup_type,
        chapter=session.chapter,
    )
    if session.student_key and session.student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_voice_turn

        emit_voice_turn(
            student_user_id=int(session.student_key),
            is_user=True,
            text=question,
            subject_name=session.subject_name,
            chapter=session.chapter,
            understanding_scores=understanding_payload,
        )
        emit_voice_turn(
            student_user_id=int(session.student_key),
            is_user=False,
            text=full_answer[:300],
            subject_name=session.subject_name,
            chapter=session.chapter,
            understanding_scores=understanding_payload,
        )
    session.tutor_state = next_tutor_state(
        current=current_state,
        scores=scores,
        assistant_reply=full_answer,
    ).value
    await _send(ws, {"type": "tutor_state", "state": session.tutor_state})


@ws_router.websocket("/ws/voice")
async def voice_ws(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    await websocket.accept()

    logger.info("Voice WS connected")

    session = _Session()
    stop = asyncio.Event()
    gen_task: asyncio.Task | None = None

    if token:
        try:
            from app.core.security import decode_token
            payload = decode_token(token)
            session.student_key = str(payload.get("sub") or "")
        except Exception:
            pass

    async def _cancel_gen() -> None:
        nonlocal gen_task
        if gen_task and not gen_task.done():
            stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(gen_task), timeout=VOICE_INTERRUPT_CANCEL_SEC)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                gen_task.cancel()
                try:
                    await gen_task
                except (asyncio.CancelledError, Exception):
                    pass
        stop.clear()
        gen_task = None

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=120)
            except asyncio.TimeoutError:
                await _send(websocket, {"type": "ping"})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type", "")

            if mtype == "session_start":
                session.configure(msg)
                from app.services.voice_performance import (
                    schedule_learner_profile_load,
                    schedule_session_prewarm,
                )

                schedule_session_prewarm(session)
                schedule_learner_profile_load(session)
                if msg.get("greet"):
                    asyncio.create_task(
                        _play_session_greeting(websocket, session),
                        name="voice-greeting",
                    )
                else:
                    await _send(websocket, {"type": "listening"})

            elif mtype == "question":
                text = postprocess_voice_transcript(
                    (msg.get("text") or "").strip(),
                    subject_name=session.subject_name,
                )
                if not text or len(text) < 2:
                    continue
                audio_b64 = str(msg.get("utterance_audio_b64") or "").strip()
                if audio_b64 and session.voice_session_id:
                    try:
                        from app.services.voice_session_profile import (
                            bootstrap_session_voice,
                            evaluate_session_user_audio,
                        )

                        audio_bytes = base64.b64decode(audio_b64)
                        # Verify against the session voiceprint (once it has enough
                        # samples) before trusting this audio as real student speech —
                        # catches tutor-echo that slipped past text-similarity matching.
                        # No-ops (accept=True) until the profile has ~1.2s of speech.
                        speaker_check = evaluate_session_user_audio(
                            session.voice_session_id, audio_bytes
                        )
                        if not speaker_check.get("accept", True):
                            from app.services import voice_protection_metrics as metrics

                            metrics.incr("speaker_rejections")
                            metrics.log_event(
                                "SPEAKER_REJECTED",
                                reason=speaker_check.get("reason"),
                                similarity=speaker_check.get("speaker_similarity"),
                                source="turn",
                            )
                            logger.debug(
                                "Turn audio rejected by session speaker check: %s",
                                speaker_check.get("reason"),
                            )
                            await _send(websocket, {"type": "listening"})
                            continue
                        # Barge/interrupt audio overlaps tutor TTS playback and
                        # carries a much higher risk of being speaker leakage —
                        # never let it train the voiceprint.
                        if not bool(msg.get("is_barge")):
                            bootstrap_session_voice(session.voice_session_id, audio_bytes)
                    except Exception as exc:
                        logger.debug("Session voice bootstrap skipped: %s", exc)
                await _cancel_gen()
                stop.clear()
                gen_task = asyncio.create_task(
                    _stream_answer(websocket, text, session, stop),
                    name="voice-gen",
                )

            elif mtype == "set_voice":
                picked = voice_for_gender(msg.get("voice_gender"), msg.get("tts_voice"))
                if picked is not None:
                    session.tts_voice = picked

            elif mtype == "interrupt":
                stop.set()
                from app.services.voice_conversation import emit_interrupt_listening

                session.tutor_state = "LISTENING"
                await emit_interrupt_listening(_send, websocket)
                await _cancel_gen()

            elif mtype == "stop":
                await _cancel_gen()
                break

            elif mtype == "session_end":
                if session.voice_session_id:
                    from app.services.voice_session_profile import clear_session_voice

                    clear_session_voice(session.voice_session_id)
                    session.voice_session_id = ""
                await _cancel_gen()
                break

            elif mtype == "ping":
                await _send(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Voice WS disconnected")
    except Exception as exc:
        logger.error("Voice WS fatal error: %s", exc, exc_info=True)
        try:
            await _send(websocket, {"type": "error", "message": VOICE_SERVER_ERROR})
        except Exception:
            pass
    finally:
        await _cancel_gen()
        if session.voice_session_id:
            from app.services.voice_session_profile import clear_session_voice

            clear_session_voice(session.voice_session_id)
        logger.info("Voice WS session ended")

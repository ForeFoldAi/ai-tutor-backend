"""
Voice REST endpoints (legacy fallback layer).

Primary interface:  WebSocket at /ws/voice  (see voice_ws.py)
Fallback interface: POST /auth/voice-stream  (binary framed stream, MP3 audio)
"""

from __future__ import annotations

import json
import logging
import re
import struct
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.edge_tts_service import (
    FALLBACK_VOICE,
    PRIMARY_VOICE,
    iter_edge_tts_mp3,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="")

_FRAME_TEXT = 1
_FRAME_AUDIO = 2  # MP3 chunk (audio/mpeg)
_FRAME_DONE = 3
_FRAME_IMAGES = 4


def _frame(ftype: int, data: bytes) -> bytes:
    return struct.pack("<BI", ftype, len(data)) + data


def _split_sentences(buf: str) -> tuple[list[str], str]:
    parts = re.split(r"(?<=[.?!])\s+", buf)
    if len(parts) > 1:
        return [s.strip() for s in parts[:-1] if s.strip()], parts[-1]
    return [], buf


async def _stream_mp3_frames(text: str) -> AsyncIterator[bytes]:
    async for chunk in iter_edge_tts_mp3(text):
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
        raise HTTPException(status_code=400, detail="Empty message")

    return StreamingResponse(
        _chat_voice_mp3_stream(text),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'inline; filename="voice.mp3"',
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


class ChapterVoiceRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    board: str = ""
    class_level: str = ""
    subject_name: str = ""
    chapter_ids: list[str] | None = None
    chapter: str = ""
    chapter_names: list[str] | None = None


@router.post("/auth/voice-stream")
async def chapter_voice_stream(req: ChapterVoiceRequest):
    """
    Chapter-aware voice stream (POST fallback when WebSocket is unavailable).

    Binary frames:
        FRAME_TEXT   (1) – UTF-8 text token
        FRAME_AUDIO  (2) – MP3 chunk
        FRAME_IMAGES (4) – JSON related images
        FRAME_DONE   (3) – end of stream
    """
    from app.core.cache import deserialize_tutor_cache, get_cached_answer
    from app.services.chat_service import chapter_aware_qa_stream

    collection = f"{req.board}_{req.class_level}_{req.subject_name}".replace(" ", "_")

    async def generate() -> AsyncIterator[bytes]:
        if req.board and req.subject_name:
            cached = await get_cached_answer(collection, req.chapter_ids, req.message)
            if cached:
                answer, imgs = deserialize_tutor_cache(cached)
                yield _frame(_FRAME_IMAGES, json.dumps(imgs).encode("utf-8"))
                for word in answer.split():
                    yield _frame(_FRAME_TEXT, (word + " ").encode())
                for part in re.split(r"(?<=[.?!])\s+", answer):
                    if not part.strip():
                        continue
                    async for framed in _stream_mp3_frames(part.strip()):
                        yield framed
                yield _frame(_FRAME_DONE, b"")
                return

        sentence_buf = ""
        side_frames: list[bytes] = []

        async def emit_imgs(imgs: list[dict]) -> None:
            side_frames.append(_frame(_FRAME_IMAGES, json.dumps(imgs).encode("utf-8")))

        async for token in chapter_aware_qa_stream(
            req.message,
            collection_name=collection,
            chapter_ids=req.chapter_ids,
            class_level=req.class_level,
            board=req.board,
            subject_name=req.subject_name,
            chapter=req.chapter,
            chapter_names=req.chapter_names,
            emit_related_images=emit_imgs,
            voice_mode=True,
        ):
            while side_frames:
                yield side_frames.pop(0)
            yield _frame(_FRAME_TEXT, token.encode("utf-8"))
            sentence_buf += token

            sentences, sentence_buf = _split_sentences(sentence_buf)
            for s in sentences:
                async for framed in _stream_mp3_frames(s):
                    yield framed

        while side_frames:
            yield side_frames.pop(0)

        if sentence_buf.strip():
            async for framed in _stream_mp3_frames(sentence_buf.strip()):
                yield framed

        yield _frame(_FRAME_DONE, b"")

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/voice/tts-info")
def tts_info() -> dict[str, str]:
    return {
        "provider": "edge-tts",
        "primary_voice": PRIMARY_VOICE,
        "fallback_voice": FALLBACK_VOICE,
        "audio_format": "audio/mpeg",
    }

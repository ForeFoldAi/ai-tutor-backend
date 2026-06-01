"""
Edge TTS streaming for the AI Tutor voice stack.

Primary voice: en-IN-NeerjaNeural
Fallback:      en-IN-PrabhatNeural
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import edge_tts
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)

PRIMARY_VOICE = "en-IN-NeerjaNeural"
FALLBACK_VOICE = "en-IN-PrabhatNeural"

_resolved_voice: str | None = None
_voice_lock = asyncio.Lock()


async def resolve_voice() -> str:
    """Return a working Indian English neural voice (cached after first probe)."""
    global _resolved_voice
    if _resolved_voice:
        return _resolved_voice

    async with _voice_lock:
        if _resolved_voice:
            return _resolved_voice

        for candidate in (PRIMARY_VOICE, FALLBACK_VOICE):
            try:
                communicate = edge_tts.Communicate("Hello", voice=candidate)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio" and chunk.get("data"):
                        _resolved_voice = candidate
                        logger.info("Edge TTS active voice: %s", candidate)
                        return candidate
            except Exception as exc:
                logger.warning("Edge TTS voice probe failed for %s: %s", candidate, exc)

        _resolved_voice = FALLBACK_VOICE
        logger.warning("Edge TTS using fallback voice without probe: %s", FALLBACK_VOICE)
        return _resolved_voice


async def stream_edge_tts(
    text: str,
    websocket: WebSocket,
    stop_event: asyncio.Event,
    *,
    voice: str | None = None,
    timing: Any | None = None,
) -> bool:
    """
    Stream MP3 chunks from edge-tts as WebSocket binary frames.

    Returns True when synthesis finished without interruption, False otherwise.
    """
    from app.services.tts_sanitize import sanitize_chunk_for_tts

    sentence = sanitize_chunk_for_tts(text)
    if not sentence or stop_event.is_set():
        return False

    voice_name = voice or await resolve_voice()
    if timing is not None:
        timing.mark_tts_started(sentence)

    try:
        communicate = edge_tts.Communicate(sentence, voice=voice_name)
        async for chunk in communicate.stream():
            if stop_event.is_set():
                return False
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            if chunk.get("type") != "audio":
                continue
            data = chunk.get("data")
            if not data:
                continue
            if timing is not None:
                timing.mark_first_mp3_generated()
            try:
                await websocket.send_bytes(data)
                if timing is not None:
                    timing.mark_first_chunk_sent()
            except Exception:
                return False
        return not stop_event.is_set()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Edge TTS failed for %r: %s", sentence[:48], exc)
        return False


async def iter_edge_tts_mp3(
    text: str,
    stop_event: asyncio.Event | None = None,
    *,
    voice: str | None = None,
) -> AsyncIterator[bytes]:
    """Yield MP3 byte chunks for REST framed streams and read-aloud."""
    from app.services.tts_sanitize import sanitize_chunk_for_tts

    sentence = sanitize_chunk_for_tts(text)
    if not sentence:
        return

    voice_name = voice or await resolve_voice()
    communicate = edge_tts.Communicate(sentence, voice=voice_name)
    async for chunk in communicate.stream():
        if stop_event and stop_event.is_set():
            return
        if chunk.get("type") != "audio":
            continue
        data = chunk.get("data")
        if data:
            yield data


async def synthesize_mp3(text: str, *, voice: str | None = None) -> bytes:
    """Collect a full MP3 payload for a single utterance."""
    parts: list[bytes] = []
    async for chunk in iter_edge_tts_mp3(text, voice=voice):
        parts.append(chunk)
    return b"".join(parts)


async def synthesize_mp3_paragraph(text: str) -> bytes:
    """Synthesize long text sentence-by-sentence and concatenate MP3."""
    import re

    buf = bytearray()
    for part in re.split(r"(?<=[.?!])\s+", text.strip()):
        s = part.strip()
        if not s:
            continue
        buf.extend(await synthesize_mp3(s))
    return bytes(buf)

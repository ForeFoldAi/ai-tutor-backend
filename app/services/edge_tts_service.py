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

from app.services.voice_prosody import prepare_voice_tts

logger = logging.getLogger(__name__)

PRIMARY_VOICE = "en-IN-NeerjaNeural"  # female
FALLBACK_VOICE = "en-IN-PrabhatNeural"  # male
ALLOWED_VOICES = frozenset({PRIMARY_VOICE, FALLBACK_VOICE})
_SEND_CHUNK_BYTES = 4096

_resolved_voice: str | None = None
_voice_lock = asyncio.Lock()

# edge-tts is an unofficial wrapper around Microsoft Edge's read-aloud
# websocket — it has no SLA and can stall mid-stream (no error, no more
# chunks) instead of raising. Without a bound here that hang blocks the
# whole per-turn TTS pipeline until the 120s watchdog in voice_ws.py fires.
_CHUNK_TIMEOUT_SEC = 8.0


def voice_for_gender(
    gender: str | None = None,
    voice: str | None = None,
) -> str | None:
    """Map male/female (or explicit Edge id) to Neerja/Prabhat. None → probe default."""
    if voice and voice in ALLOWED_VOICES:
        return voice
    g = (gender or "").strip().lower()
    if g in ("male", "m", "prabhat"):
        return FALLBACK_VOICE
    if g in ("female", "f", "neerja"):
        return PRIMARY_VOICE
    return None


async def _aclose_async_gen(gen: AsyncIterator) -> None:
    """Close an async generator so edge-tts aiohttp sessions are released."""
    aclose = getattr(gen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        pass


async def _iter_communicate_stream(communicate: edge_tts.Communicate) -> AsyncIterator[dict]:
    """Wrap communicate.stream() and always aclose on early exit or cancel.

    Bounds the wait for each chunk so a stalled edge-tts connection raises
    asyncio.TimeoutError instead of hanging forever (see _CHUNK_TIMEOUT_SEC).
    """
    stream = communicate.stream()
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=_CHUNK_TIMEOUT_SEC)
            except StopAsyncIteration:
                break
            yield chunk
    finally:
        await _aclose_async_gen(stream)


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
                communicate = _communicate("Hello", voice=candidate)
                async for chunk in _iter_communicate_stream(communicate):
                    if chunk["type"] == "audio" and chunk.get("data"):
                        _resolved_voice = candidate
                        logger.info("Edge TTS active voice: %s", candidate)
                        return candidate
            except Exception as exc:
                logger.warning("Edge TTS voice probe failed for %s: %s", candidate, exc)

        _resolved_voice = FALLBACK_VOICE
        logger.warning("Edge TTS using fallback voice without probe: %s", FALLBACK_VOICE)
        return _resolved_voice


def _communicate(text: str, *, voice: str, chunk_index: int = 0) -> edge_tts.Communicate:
    """Build Communicate with teaching-friendly prosody (plain text + rate/pitch/volume)."""
    spoken, rate, pitch, volume = prepare_voice_tts(text, chunk_index=chunk_index)
    if not spoken:
        spoken = " "
    return edge_tts.Communicate(
        spoken, voice=voice, rate=rate, pitch=pitch, volume=volume
    )


async def stream_edge_tts(
    text: str,
    websocket: WebSocket,
    stop_event: asyncio.Event,
    *,
    voice: str | None = None,
    timing: Any | None = None,
    chunk_index: int = 0,
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
        communicate = _communicate(sentence, voice=voice_name, chunk_index=chunk_index)
        async for chunk in _iter_communicate_stream(communicate):
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
    chunk_index: int = 0,
) -> AsyncIterator[bytes]:
    """Yield MP3 byte chunks for REST framed streams and read-aloud."""
    from app.services.tts_sanitize import sanitize_chunk_for_tts

    sentence = sanitize_chunk_for_tts(text)
    if not sentence:
        return

    voice_name = voice or await resolve_voice()
    communicate = _communicate(sentence, voice=voice_name, chunk_index=chunk_index)
    async for chunk in _iter_communicate_stream(communicate):
        if stop_event and stop_event.is_set():
            return
        if chunk.get("type") != "audio":
            continue
        data = chunk.get("data")
        if data:
            yield data


async def synthesize_mp3(
    text: str,
    *,
    voice: str | None = None,
    stop_event: asyncio.Event | None = None,
    chunk_index: int = 0,
) -> bytes:
    """Collect a full MP3 payload for a single speech unit (prefetch path)."""
    parts: list[bytes] = []
    async for chunk in iter_edge_tts_mp3(
        text, stop_event=stop_event, voice=voice, chunk_index=chunk_index
    ):
        parts.append(chunk)
    return b"".join(parts)


async def send_mp3_bytes(
    data: bytes,
    websocket: WebSocket,
    stop_event: asyncio.Event,
    *,
    timing: Any | None = None,
) -> bool:
    """Stream a prefetched MP3 buffer as WebSocket binary frames."""
    if not data or stop_event.is_set():
        return False
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        for i in range(0, len(data), _SEND_CHUNK_BYTES):
            if stop_event.is_set():
                return False
            chunk = data[i : i + _SEND_CHUNK_BYTES]
            if timing is not None and timing.first_mp3_generated_at is None:
                timing.mark_first_mp3_generated()
            await websocket.send_bytes(chunk)
            if timing is not None and timing.first_chunk_sent_at is None:
                timing.mark_first_chunk_sent()
        return not stop_event.is_set()
    except Exception:
        return False


async def synthesize_mp3_legacy(text: str, *, voice: str | None = None) -> bytes:
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
        buf.extend(await synthesize_mp3_legacy(s))
    return bytes(buf)

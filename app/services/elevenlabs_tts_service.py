"""
ElevenLabs streaming TTS — premium production voice provider.

Same call surface as edge_tts_service (resolve_voice, voice_for_gender,
stream_tts, synthesize_mp3) so app.services.tts_provider can select between
providers without touching call sites in voice_ws.py / voice_tts_orchestrator.py.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from app.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_VOICE_ID_FEMALE,
    ELEVENLABS_VOICE_ID_MALE,
)
from app.services.voice_prosody import apply_voice_prosody

logger = logging.getLogger(__name__)

PRIMARY_VOICE = ELEVENLABS_VOICE_ID_FEMALE
FALLBACK_VOICE = ELEVENLABS_VOICE_ID_MALE
ALLOWED_VOICES = frozenset({PRIMARY_VOICE, FALLBACK_VOICE})

_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)


def voice_for_gender(gender: str | None = None, voice: str | None = None) -> str | None:
    """Map male/female (or an explicit ElevenLabs voice id) to Rachel/Adam. None → default."""
    if voice and voice in ALLOWED_VOICES:
        return voice
    g = (gender or "").strip().lower()
    if g in ("male", "m"):
        return FALLBACK_VOICE
    if g in ("female", "f"):
        return PRIMARY_VOICE
    return None


async def resolve_voice() -> str:
    """No probe needed — ElevenLabs voice ids are static and always usable."""
    return PRIMARY_VOICE


async def _iter_elevenlabs_mp3(text: str, *, voice: str) -> AsyncIterator[bytes]:
    url = f"{_API_BASE}/{voice}/stream?output_format=mp3_44100_128"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"ElevenLabs TTS {resp.status_code}: {body[:200]!r}")
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


async def iter_tts_mp3(
    text: str,
    stop_event: Any | None = None,
    *,
    voice: str | None = None,
    chunk_index: int = 0,
) -> AsyncIterator[bytes]:
    """Yield MP3 byte chunks for REST framed streams and read-aloud."""
    sentence = apply_voice_prosody(text, chunk_index=chunk_index)
    if not sentence:
        return
    voice_id = voice or await resolve_voice()
    async for chunk in _iter_elevenlabs_mp3(sentence, voice=voice_id):
        if stop_event and stop_event.is_set():
            return
        yield chunk


async def synthesize_mp3(
    text: str,
    *,
    voice: str | None = None,
    stop_event: Any | None = None,
    chunk_index: int = 0,
) -> bytes:
    """Collect a full MP3 payload for a single speech unit (prefetch path)."""
    parts: list[bytes] = []
    async for chunk in iter_tts_mp3(text, stop_event=stop_event, voice=voice, chunk_index=chunk_index):
        parts.append(chunk)
    return b"".join(parts)


async def stream_tts(
    text: str,
    websocket: WebSocket,
    stop_event: Any,
    *,
    voice: str | None = None,
    timing: Any | None = None,
    chunk_index: int = 0,
) -> bool:
    """Stream MP3 chunks from ElevenLabs as WebSocket binary frames.

    Returns True when synthesis finished without interruption, False otherwise.
    """
    from app.services.tts_sanitize import sanitize_chunk_for_tts

    sentence = sanitize_chunk_for_tts(text)
    if not sentence or stop_event.is_set():
        return False

    voice_id = voice or await resolve_voice()
    if timing is not None:
        timing.mark_tts_started(sentence)

    try:
        first_chunk = True
        async for data in iter_tts_mp3(sentence, stop_event=stop_event, voice=voice_id, chunk_index=chunk_index):
            if stop_event.is_set():
                return False
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            if first_chunk and timing is not None:
                timing.mark_first_mp3_generated()
                first_chunk = False
            try:
                await websocket.send_bytes(data)
                if timing is not None:
                    timing.mark_first_chunk_sent()
            except Exception:
                return False
        return not stop_event.is_set()
    except Exception as exc:
        logger.warning("ElevenLabs TTS failed for %r: %s", sentence[:48], exc)
        return False

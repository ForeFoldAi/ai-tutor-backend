"""
TTS provider selection — picks ElevenLabs (premium) or edge-tts (free
fallback) based on VOICE_TTS_PROVIDER, so voice_ws.py / voice_tts_orchestrator.py /
voice_http_tts.py call one stable surface regardless of backend.

When ElevenLabs is the selected provider, every call still falls back to
edge-tts per-chunk on failure (quota exhausted, plan restriction like the
"paid_plan_required" 402 free accounts hit on library voices, network) — a
turn must never go silent just because the premium provider is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.config import VOICE_TTS_PROVIDER
from app.services import edge_tts_service as _edge
from app.services import elevenlabs_tts_service as _eleven
from app.services.edge_tts_service import send_mp3_bytes  # provider-agnostic byte framing

logger = logging.getLogger(__name__)


def _interrupted(stop_event: Any) -> bool:
    return bool(stop_event and stop_event.is_set())


async def _synthesize_mp3_with_fallback(
    text: str, *, voice: str | None = None, stop_event: Any | None = None, chunk_index: int = 0
) -> bytes:
    """Try ElevenLabs; fall back to edge-tts on any failure (quota, plan
    restriction, network) rather than leaving the turn silent — an
    ElevenLabs voice id is meaningless to edge-tts, so the fallback call
    always uses its own default voice, not `voice`."""
    data = await _eleven.synthesize_mp3(text, voice=voice, stop_event=stop_event, chunk_index=chunk_index)
    if data or _interrupted(stop_event):
        return data
    logger.warning("ElevenLabs TTS unavailable, falling back to edge-tts for this chunk")
    return await _edge.synthesize_mp3(text, voice=None, stop_event=stop_event, chunk_index=chunk_index)


async def _stream_tts_with_fallback(
    text: str,
    websocket: Any,
    stop_event: Any,
    *,
    voice: str | None = None,
    timing: Any | None = None,
    chunk_index: int = 0,
) -> bool:
    ok = await _eleven.stream_tts(text, websocket, stop_event, voice=voice, timing=timing, chunk_index=chunk_index)
    if ok or _interrupted(stop_event):
        return ok
    logger.warning("ElevenLabs TTS unavailable, falling back to edge-tts for this chunk")
    return await _edge.stream_edge_tts(
        text, websocket, stop_event, voice=None, timing=timing, chunk_index=chunk_index
    )


async def _iter_tts_mp3_with_fallback(
    text: str, stop_event: Any | None = None, *, voice: str | None = None, chunk_index: int = 0
) -> AsyncIterator[bytes]:
    got_any = False
    try:
        async for chunk in _eleven.iter_tts_mp3(text, stop_event=stop_event, voice=voice, chunk_index=chunk_index):
            got_any = True
            yield chunk
    except Exception as exc:
        logger.warning("ElevenLabs TTS iter failed: %s", exc)
    if got_any or _interrupted(stop_event):
        return
    logger.warning("ElevenLabs TTS unavailable, falling back to edge-tts for this stream")
    async for chunk in _edge.iter_edge_tts_mp3(text, stop_event=stop_event, voice=None, chunk_index=chunk_index):
        yield chunk


if VOICE_TTS_PROVIDER == "elevenlabs":
    from app.services.elevenlabs_tts_service import FALLBACK_VOICE, PRIMARY_VOICE, resolve_voice, voice_for_gender

    synthesize_mp3 = _synthesize_mp3_with_fallback
    stream_tts = _stream_tts_with_fallback
    iter_tts_mp3 = _iter_tts_mp3_with_fallback

    logger.info("Voice TTS provider: elevenlabs (falls back to edge-tts per-chunk on failure)")
else:
    from app.services.edge_tts_service import (
        FALLBACK_VOICE,
        PRIMARY_VOICE,
        iter_edge_tts_mp3 as iter_tts_mp3,
        resolve_voice,
        stream_edge_tts as stream_tts,
        synthesize_mp3,
        voice_for_gender,
    )

    logger.info("Voice TTS provider: edge (set ELEVENLABS_API_KEY for premium voice)")

__all__ = [
    "FALLBACK_VOICE",
    "PRIMARY_VOICE",
    "iter_tts_mp3",
    "resolve_voice",
    "send_mp3_bytes",
    "stream_tts",
    "synthesize_mp3",
    "voice_for_gender",
]

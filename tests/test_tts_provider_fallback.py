"""
A turn must never go silent just because ElevenLabs is unavailable — these
test the exact failure mode reported in production: a free ElevenLabs plan
returns 402 payment_required for library voices on every call. The fallback
functions are unconditionally importable (not gated behind VOICE_TTS_PROVIDER)
specifically so this stays testable regardless of which provider is active
in the current environment's .env.
"""

import asyncio

from app.services.tts_provider import (
    _iter_tts_mp3_with_fallback,
    _stream_tts_with_fallback,
    _synthesize_mp3_with_fallback,
)

_PAYMENT_REQUIRED = RuntimeError(
    'ElevenLabs TTS 402: {"detail":{"type":"payment_required","code":"paid_plan_required",'
    '"message":"Free users cannot use library voices via the API."}}'
)


def test_synthesize_mp3_falls_back_to_edge_on_402(monkeypatch):
    import app.services.tts_provider as tp

    async def _eleven_fails(text, *, voice, stop_event=None, chunk_index=0):
        return b""  # elevenlabs_tts_service.synthesize_mp3 degrades failures to b""

    async def _edge_ok(text, *, voice, stop_event=None, chunk_index=0):
        assert voice is None  # an ElevenLabs voice id must never reach edge-tts
        return b"edge-mp3-bytes"

    monkeypatch.setattr(tp._eleven, "synthesize_mp3", _eleven_fails)
    monkeypatch.setattr(tp._edge, "synthesize_mp3", _edge_ok)

    data = asyncio.run(_synthesize_mp3_with_fallback("Okay, picture this.", voice="21m00Tcm4TlvDq8ikWAM"))
    assert data == b"edge-mp3-bytes"


def test_stream_tts_falls_back_to_edge_on_402(monkeypatch):
    import app.services.tts_provider as tp

    async def _eleven_fails(text, websocket, stop_event, *, voice, timing=None, chunk_index=0):
        return False

    async def _edge_ok(text, websocket, stop_event, *, voice, timing=None, chunk_index=0):
        assert voice is None
        return True

    monkeypatch.setattr(tp._eleven, "stream_tts", _eleven_fails)
    monkeypatch.setattr(tp._edge, "stream_edge_tts", _edge_ok)

    stop = asyncio.Event()
    ok = asyncio.run(
        _stream_tts_with_fallback("Okay, picture this.", object(), stop, voice="21m00Tcm4TlvDq8ikWAM")
    )
    assert ok is True


def test_stream_tts_does_not_fall_back_when_interrupted(monkeypatch):
    """A cancelled turn must not trigger a wasted fallback call."""
    import app.services.tts_provider as tp

    async def _eleven_fails(text, websocket, stop_event, *, voice, timing=None, chunk_index=0):
        return False

    async def _edge_should_not_run(*a, **k):
        raise AssertionError("fallback must not run when the turn was interrupted")

    monkeypatch.setattr(tp._eleven, "stream_tts", _eleven_fails)
    monkeypatch.setattr(tp._edge, "stream_edge_tts", _edge_should_not_run)

    stop = asyncio.Event()
    stop.set()
    ok = asyncio.run(_stream_tts_with_fallback("text", object(), stop, voice="x"))
    assert ok is False


def test_iter_tts_mp3_falls_back_when_elevenlabs_raises(monkeypatch):
    """elevenlabs_tts_service.iter_tts_mp3 raises rather than degrading
    silently (matches edge_tts_service's own iter_edge_tts_mp3 behavior) —
    the wrapper must catch that and still deliver audio via edge-tts."""
    import app.services.tts_provider as tp

    async def _eleven_raises(text, *, stop_event=None, voice=None, chunk_index=0):
        raise _PAYMENT_REQUIRED
        yield b""  # pragma: no cover - unreachable, keeps this an async generator

    async def _edge_ok(text, *, stop_event=None, voice=None, chunk_index=0):
        assert voice is None
        yield b"edge-chunk-1"
        yield b"edge-chunk-2"

    monkeypatch.setattr(tp._eleven, "iter_tts_mp3", _eleven_raises)
    monkeypatch.setattr(tp._edge, "iter_edge_tts_mp3", _edge_ok)

    async def _collect():
        return [c async for c in _iter_tts_mp3_with_fallback("Okay, picture this.", voice="21m00Tcm4TlvDq8ikWAM")]

    chunks = asyncio.run(_collect())
    assert chunks == [b"edge-chunk-1", b"edge-chunk-2"]

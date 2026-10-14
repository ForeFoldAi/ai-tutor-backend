"""ponytail: ElevenLabs premium TTS provider — voice mapping + failure handling."""

import asyncio

import pytest

from app.services.elevenlabs_tts_service import (
    FALLBACK_VOICE,
    PRIMARY_VOICE,
    stream_tts,
    synthesize_mp3,
    voice_for_gender,
)


def test_voice_for_gender_female():
    assert voice_for_gender("female") == PRIMARY_VOICE
    assert voice_for_gender("f") == PRIMARY_VOICE


def test_voice_for_gender_male():
    assert voice_for_gender("male") == FALLBACK_VOICE
    assert voice_for_gender("m") == FALLBACK_VOICE


def test_voice_for_gender_explicit_override():
    assert voice_for_gender("male", PRIMARY_VOICE) == PRIMARY_VOICE


def test_voice_for_gender_unknown_is_none():
    assert voice_for_gender(None) is None
    assert voice_for_gender("other") is None


def test_synthesize_mp3_propagates_provider_failure(monkeypatch):
    """synthesize_mp3 does not swallow errors itself (matches edge_tts_service) —
    callers (prefetch slot, orchestrator live-pull) are the ones that must catch
    this and degrade to empty bytes rather than crashing the turn."""
    import app.services.elevenlabs_tts_service as svc

    async def _boom(text, *, voice):
        raise RuntimeError("simulated network stall")
        yield b""  # pragma: no cover - unreachable, makes this an async generator

    monkeypatch.setattr(svc, "_iter_elevenlabs_mp3", _boom)

    with pytest.raises(RuntimeError):
        asyncio.run(synthesize_mp3("hello world", voice=PRIMARY_VOICE))


def test_stream_tts_returns_false_on_provider_failure(monkeypatch):
    import app.services.elevenlabs_tts_service as svc

    class _FakeWebSocket:
        client_state = "CONNECTED"

    async def _boom(text, *, voice):
        raise RuntimeError("simulated network stall")
        yield b""  # pragma: no cover

    monkeypatch.setattr(svc, "_iter_elevenlabs_mp3", _boom)
    monkeypatch.setattr(
        "app.services.elevenlabs_tts_service.WebSocketState",
        type("WS", (), {"CONNECTED": "CONNECTED"}),
    )

    stop = asyncio.Event()
    ok = asyncio.run(stream_tts("Hello there.", _FakeWebSocket(), stop, voice=PRIMARY_VOICE))
    assert ok is False

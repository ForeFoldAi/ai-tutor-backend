"""TTS prosody — plain text + Communicate rate/pitch (no inline SSML)."""

from __future__ import annotations


def test_prosody_disabled_passthrough(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", False)
    text, rate, pitch = vp.prepare_voice_tts("Hello there", chunk_index=1)
    assert text == "Hello there"
    assert rate == vp.VOICE_TTS_RATE
    assert pitch == vp.VOICE_TTS_PITCH
    assert vp.apply_voice_prosody("Hello there") == "Hello there"


def test_prosody_math_slower_no_ssml(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    text, rate, pitch = vp.prepare_voice_tts(
        "Step 2: multiply both sides by 3",
        chunk_index=1,
    )
    assert text == "Step 2: multiply both sides by 3"
    assert "<speak>" not in text
    assert rate == "-6%"
    assert pitch == "+1Hz"

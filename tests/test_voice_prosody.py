"""TTS prosody — plain text + Communicate rate/pitch/volume (no inline SSML)."""

from __future__ import annotations


def test_prosody_disabled_passthrough(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", False)
    text, rate, pitch, volume = vp.prepare_voice_tts("Hello there", chunk_index=1)
    assert text == "Hello there"
    assert rate == vp.VOICE_TTS_RATE
    assert pitch == vp.VOICE_TTS_PITCH
    assert volume == vp.VOICE_TTS_VOLUME
    assert vp.apply_voice_prosody("Hello there") == "Hello there"


def test_prosody_math_step_slightly_slower_no_ssml(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+8%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")
    text, rate, pitch, volume = vp.prepare_voice_tts(
        "Step 2: multiply both sides by 3",
        chunk_index=1,
    )
    assert text == "Step 2: multiply both sides by 3"
    assert "<speak>" not in text
    assert rate == "+6%"  # 8 - 2 math step only
    assert pitch == "+0Hz"
    assert volume == "+0%"


def test_prosody_explanation_keeps_base_rate(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+10%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")

    _, rate_dot, pitch_dot, _ = vp.prepare_voice_tts("Plants need sunlight.", chunk_index=2)
    assert rate_dot == "+10%"
    assert pitch_dot == "+0Hz"


def test_prosody_question_tiny_pitch_not_cartoon(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+10%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")
    _, rate, pitch, volume = vp.prepare_voice_tts("Does that make sense?", chunk_index=1)
    assert rate == "+10%"
    assert pitch == "+1Hz"
    assert volume == "+0%"


def test_prosody_short_ack_brighter(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")
    _, rate, pitch, volume = vp.prepare_voice_tts("Great!", chunk_index=1)
    assert rate == "+1%"
    assert pitch == "+1Hz"
    assert volume == "+1%"


def test_prosody_opening_chunk_warmer(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")
    _, rate, pitch, volume = vp.prepare_voice_tts("Today we look at maps", chunk_index=0)
    assert rate in ("+0%", "-1%")
    assert pitch == "+1Hz"
    assert volume == "+1%"


def test_prosody_mid_turn_intro_no_opening_boost(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+8%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")
    _, rate, pitch, volume = vp.prepare_voice_tts("Today we look at maps", chunk_index=1)
    assert rate in ("+7%", "+8%")
    assert pitch == "+0Hz"
    assert volume == "+0%"


def test_everyday_textbook_noun_does_not_slow_everything(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    _, rate, pitch, _ = vp.prepare_voice_tts(
        "Natural resources and weather shape how people live.",
        chunk_index=1,
    )
    assert rate == "+0%"
    assert pitch == "+0Hz"

"""ponytail: gender → Edge TTS voice mapping."""

from app.services.edge_tts_service import (
    FALLBACK_VOICE,
    PRIMARY_VOICE,
    voice_for_gender,
)


def test_voice_for_gender_female_neerja():
    assert voice_for_gender("female") == PRIMARY_VOICE
    assert voice_for_gender("f") == PRIMARY_VOICE
    assert voice_for_gender("neerja") == PRIMARY_VOICE


def test_voice_for_gender_male_prabhat():
    assert voice_for_gender("male") == FALLBACK_VOICE
    assert voice_for_gender("m") == FALLBACK_VOICE
    assert voice_for_gender("prabhat") == FALLBACK_VOICE


def test_voice_for_gender_explicit_overrides():
    assert voice_for_gender("male", PRIMARY_VOICE) == PRIMARY_VOICE
    assert voice_for_gender(None, FALLBACK_VOICE) == FALLBACK_VOICE


def test_voice_for_gender_unknown_is_none():
    assert voice_for_gender(None) is None
    assert voice_for_gender("") is None
    assert voice_for_gender("other") is None
    assert voice_for_gender(None, "en-US-FakeNeural") is None

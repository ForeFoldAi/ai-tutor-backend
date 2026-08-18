"""Wind TTS respell is Communicate-only; UI text stays 'wind'."""

from app.services.tts_sanitize import sanitize_chunk_for_tts
from app.services.voice_pronunciation import prepare_text_for_tts
from app.services.voice_prosody import apply_voice_prosody, prepare_speech_delivery, prepare_voice_tts


def test_air_uses_wihnd_not_whind():
    assert prepare_text_for_tts("The wind is blowing.") == "The wihnd is blowing."
    assert prepare_text_for_tts("The wind is very strong today.") == (
        "The wihnd is very strong today."
    )
    assert prepare_text_for_tts(
        "Wind energy is an important source of renewable energy."
    ).startswith("Wihnd ")
    assert prepare_text_for_tts("The wind speed increased.") == "The wihnd speed increased."
    spoken, *_ = prepare_voice_tts("The wind is blowing.", chunk_index=1)
    assert spoken == "The wihnd is blowing."
    assert "whind" not in spoken.lower()


def test_verb_keeps_wind():
    assert prepare_text_for_tts("Wind the clock.") == "Wind the clock."
    assert prepare_text_for_tts("Wind the rope around the pole.") == (
        "Wind the rope around the pole."
    )
    assert prepare_text_for_tts("Wind it up.") == "Wind it up."
    spoken, *_ = prepare_voice_tts("Wind the clock.", chunk_index=1)
    assert spoken == "Wind the clock."


def test_window_untouched():
    assert prepare_text_for_tts("Open the window.") == "Open the window."
    assert prepare_text_for_tts("A winding road.") == "A winding road."


def test_ui_and_sanitize_keep_wind():
    src = "The wind is blowing."
    assert sanitize_chunk_for_tts(src) == src
    assert apply_voice_prosody(src) == src
    assert prepare_speech_delivery(src).text == src

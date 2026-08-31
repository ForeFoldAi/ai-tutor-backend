"""Tests for voice STT post-processing."""

from app.services.voice_stt_postprocess import postprocess_voice_transcript


def test_photo_synthesis_fix():
    assert postprocess_voice_transcript("what is photo synthesis") == "what is photosynthesis"


def test_multiplication_fix():
    assert postprocess_voice_transcript("explain multiply cation") == "explain multiplication"


def test_math_into_times():
    assert postprocess_voice_transcript("5 into 3", subject_name="Mathematics") == "5 * 3"


def test_math_spoken_operators():
    out = postprocess_voice_transcript(
        "solve 2x plus 5 equals 15",
        subject_name="Mathematics",
    )
    assert out == "solve 2x + 5 = 15"
    assert (
        postprocess_voice_transcript(
            "what is 20 percent of 500",
            subject_name="Mathematics",
        )
        == "what is 20 % of 500"
    )


def test_word_typo_fix():
    assert postprocess_voice_transcript("photosynthisis") == "photosynthesis"


def test_empty_passthrough():
    assert postprocess_voice_transcript("") == ""
    assert postprocess_voice_transcript("  hello  ") == "hello"

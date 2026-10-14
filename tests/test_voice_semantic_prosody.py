"""Semantic teacher prosody — intent, chunks, sanitize (no live Edge-TTS)."""

from app.services.tts_sanitize import sanitize_for_tts
from app.services.voice_chunking import extract_voice_chunks
from app.services.voice_prosody import (
    SpeechIntent,
    classify_speech_intent,
    prepare_speech_delivery,
)


def _hz(pitch: str) -> int:
    return int(pitch.replace("Hz", "").replace("+", "") or "0")


def test_greeting_warm_not_quiz():
    assert classify_speech_intent("Hi! How are you today?") == SpeechIntent.GREETING
    d = prepare_speech_delivery("Hi! How are you today?", chunk_index=0)
    assert d.speech_intent == SpeechIntent.GREETING
    assert _hz(d.pitch) <= 2


def test_definition_slightly_slower(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    text = "A kingdom is a territory ruled by a king or queen."
    assert classify_speech_intent(text) == SpeechIntent.DEFINITION
    d = prepare_speech_delivery(text, chunk_index=1)
    assert d.rate == "-2%"
    assert d.pitch == "+0Hz"


def test_important_point_subtle_emphasis(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    text = "The most important point is that kingdoms had their own rulers."
    d = prepare_speech_delivery(text, chunk_index=1)
    assert d.speech_intent == SpeechIntent.IMPORTANT_POINT
    assert d.rate == "-3%"
    assert _hz(d.pitch) <= 1


def test_example_stays_conversational(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    text = "For example, the Vijayanagara Empire was a powerful kingdom."
    d = prepare_speech_delivery(text, chunk_index=1)
    assert d.speech_intent == SpeechIntent.EXAMPLE
    assert d.rate == "+0%"
    assert d.pitch == "+0Hz"


def test_question_keeps_mark_tiny_pitch(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    text = "Why do you think kingdoms became powerful?"
    d = prepare_speech_delivery(text, chunk_index=1)
    assert d.speech_intent == SpeechIntent.QUESTION
    assert d.text.endswith("?")
    assert d.pitch == "+1Hz"


def test_clarification_follow_up():
    text = "Yes, let me explain that in a simpler way."
    assert classify_speech_intent(text) == SpeechIntent.CLARIFICATION


def test_lesson_introduction():
    text = (
        "Sure! Let's learn about this lesson. "
        "It explains how India's political map changed over time."
    )
    assert classify_speech_intent(text) == SpeechIntent.INTRODUCTION


def test_summary_calm(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    text = "So, in simple words, a kingdom was a territory governed by its own ruler."
    d = prepare_speech_delivery(text, chunk_index=1)
    assert d.speech_intent == SpeechIntent.SUMMARY
    assert d.rate == "-2%"


def test_encouragement():
    assert classify_speech_intent("That's a good question!") == SpeechIntent.ENCOURAGEMENT


def test_long_explanation_splits_into_few_units_not_dozens():
    text = (
        "Sure! Let's understand kingdoms first. "
        "A kingdom is a territory ruled by a king or queen. "
        "For example, the Vijayanagara Empire was a powerful South Indian kingdom. "
        "The most important point is that each kingdom had its own ruler. "
        "Why do you think this happened? "
        "So, in simple words, a kingdom was a territory governed by its own ruler."
    )
    chunks, rest = extract_voice_chunks(text, chunks_emitted=0)
    units = chunks + ([rest] if rest.strip() else [])
    assert 2 <= len(units) <= 8
    assert not any(len(u.split()) <= 2 for u in chunks)


def test_empty_and_short():
    assert classify_speech_intent("") == SpeechIntent.EXPLANATION
    d = prepare_speech_delivery("Ok.", chunk_index=1)
    assert d.text in ("Ok.", "Okay.", "Ok")


def test_abbreviations_not_split():
    text = (
        "Dr. A.P.J. Abdul Kalam visited the class at 10:00 a.m. "
        "He spoke about Chapter 2.1 and the value 3.14 in e.g. science class today "
        "while students listened carefully to every word of the lecture"
    )
    chunks, rest = extract_voice_chunks(text, chunks_emitted=1)
    joined = " ".join(chunks + ([rest] if rest else []))
    assert "Abdul Kalam" in joined
    if chunks:
        assert not chunks[0].rstrip().endswith("Dr.")
        assert "3.14" in joined
        assert "Chapter 2.1" in joined or "2.1" in joined


def test_markdown_cleanup_spoken_form():
    out = sanitize_for_tts(
        "## Key Points\n\n- Kingdoms had rulers.\n- They controlled territories.\n"
        "- Some kingdoms became powerful."
    )
    assert out == (
        "Key Points. Kingdoms had rulers. They controlled territories. "
        "Some kingdoms became powerful."
    ) or (
        "Key Points" in out
        and "Kingdoms had rulers" in out
        and "#" not in out
        and "-" not in out
    )

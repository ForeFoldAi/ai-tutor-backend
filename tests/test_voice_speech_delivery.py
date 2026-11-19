"""Speech delivery — intent, prosody, and semantic chunking."""

from app.services.voice_chunking import extract_voice_chunks
from app.services.voice_prosody import (
    SpeechIntent,
    classify_speech_intent,
    prepare_speech_delivery,
)


def test_classify_important_point():
    assert classify_speech_intent(
        "The most important point is that Shivaji built a strong administration."
    ) == SpeechIntent.IMPORTANT_POINT


def test_classify_question():
    assert classify_speech_intent("Now, why was this important?") == SpeechIntent.QUESTION


def test_classify_example():
    assert classify_speech_intent("Here's a simple example.") == SpeechIntent.EXAMPLE


def test_classify_summary():
    assert classify_speech_intent("So remember these three main ideas.") == SpeechIntent.SUMMARY


def test_important_point_slower_delivery(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_RATE", "+0%")
    d = prepare_speech_delivery(
        "The most important thing to remember is Shivaji's administration.",
        chunk_index=1,
    )
    assert d.speech_intent == SpeechIntent.IMPORTANT_POINT
    assert d.rate in ("-3%", "-2%", "-4%")
    assert d.pause_before_ms == 0


def test_question_rising_intonation(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    d = prepare_speech_delivery("Does that make sense?", chunk_index=1)
    assert d.speech_intent == SpeechIntent.QUESTION
    hz = int(d.pitch.replace("Hz", "").replace("+", ""))
    assert 0 <= hz <= 2


def test_opening_warmth_only_first_chunk(monkeypatch):
    from app.services import voice_prosody as vp

    monkeypatch.setattr(vp, "VOICE_SSML_PROSODY", True)
    monkeypatch.setattr(vp, "VOICE_TTS_PITCH", "+0Hz")
    monkeypatch.setattr(vp, "VOICE_TTS_VOLUME", "+0%")
    first = prepare_speech_delivery("Let's start with today's lesson.", chunk_index=0)
    second = prepare_speech_delivery("Let's start with today's lesson.", chunk_index=1)
    assert first.pitch != second.pitch or first.volume != second.volume


def test_teaching_boundary_split():
    text = (
        "Let's understand the main idea first. For example, imagine you manage a small team. "
        "Then we look at why that mattered in history class today"
    )
    chunks, rest = extract_voice_chunks(text, chunks_emitted=1)
    assert chunks
    assert any("For example" in c or "main idea" in c for c in chunks)


def test_first_unit_prefers_full_phrase_not_three_words():
    chunks, rest = extract_voice_chunks(
        "Let's understand the main idea first today",
        chunks_emitted=0,
    )
    if chunks:
        assert len(chunks[0].split()) >= 4


def test_long_answer_splits_into_multiple_units_not_one_block():
    text = (
        "Good question. The reason is political unity. "
        "Shivaji built a strong administration. "
        "For example, he organized revenue collection. "
        "Does that make sense?"
    )
    chunks, _ = extract_voice_chunks(text, chunks_emitted=0)
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) < len(text) + 20

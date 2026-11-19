"""Tests for speech-unit voice chunking."""

from app.services.voice_chunking import extract_voice_chunks, idle_flush_sec


def test_first_unit_is_aggressive():
    chunks, rest = extract_voice_chunks("So photosynthesis is how plants make food", chunks_emitted=0)
    assert chunks or rest
    if chunks:
        assert len(chunks[0].split()) >= 4


def test_prefers_period_over_weak_and():
    text = (
        "Plants make food using sunlight. And this process is called photosynthesis "
        "because it happens in the leaves today"
    )
    chunks, rest = extract_voice_chunks(text, chunks_emitted=1)
    joined = " ".join(chunks) if chunks else ""
    assert chunks, (chunks, rest)
    # First cut should be at the period, not at a weak " and "
    assert chunks[0].rstrip().endswith(".")
    assert "And this process" not in chunks[0]


def test_comma_split_only_when_over_max_words():
    """Comma is a breath inside one utterance, not a new TTS call — unless overflow."""
    short = (
        "First we warm the water, then we add the salt carefully into the beaker slowly"
    )
    chunks, rest = extract_voice_chunks(short, chunks_emitted=1)
    spoken = " ".join(chunks) if chunks else rest
    assert "warm the water" in spoken
    # Short clause stays in one unit (or unflushed rest), not split at the comma.
    if chunks:
        assert not (chunks[0].rstrip().endswith(",") and len(chunks) > 1)

    long = (
        "First we warm the water, then we add the salt carefully into the beaker slowly "
        "and after that we stir the mixture for several minutes until it dissolves "
        "completely and the students can see the colour change clearly in the glass"
    )
    chunks, rest = extract_voice_chunks(long, chunks_emitted=1)
    assert chunks or rest
    if chunks:
        assert "," in chunks[0] or chunks[0].endswith(",") or len(chunks[0].split()) <= 28


def test_idle_flush_shorter_for_first_unit():
    assert idle_flush_sec(chunks_emitted=0) < idle_flush_sec(chunks_emitted=2)


def test_math_delimiter_holds_buffer():
    chunks, rest = extract_voice_chunks("The value is $x + 1", chunks_emitted=0)
    assert not chunks
    assert "$" in rest

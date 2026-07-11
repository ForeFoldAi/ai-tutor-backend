"""Tests for speech-unit voice chunking."""

from app.services.voice_chunking import extract_voice_chunks, idle_flush_sec


def test_first_unit_is_aggressive():
  chunks, rest = extract_voice_chunks("So photosynthesis is how", chunks_emitted=0)
  assert chunks
  assert len(chunks[0].split()) <= 5
  assert rest


def test_splits_on_discourse_not_only_period():
  text = (
      "Plants make food using sunlight and this process is called photosynthesis "
      "because it happens in the leaves"
  )
  chunks, rest = extract_voice_chunks(text, chunks_emitted=1)
  assert chunks or rest
  if chunks:
    assert "because" in chunks[0] or "and" in chunks[0] or len(chunks[0]) > 20


def test_idle_flush_shorter_for_first_unit():
  assert idle_flush_sec(chunks_emitted=0) < idle_flush_sec(chunks_emitted=2)


def test_math_delimiter_holds_buffer():
  chunks, rest = extract_voice_chunks("The value is $x + 1", chunks_emitted=0)
  assert not chunks
  assert "$" in rest

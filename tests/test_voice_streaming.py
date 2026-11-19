"""Phase 2 — voice streaming queue and buffer."""

import asyncio
import time

from app.config import VOICE_SPEECH_QUEUE_MAXSIZE
from app.services.voice_chunking import VoicePipelineTiming, is_flushable_fragment
from app.services.voice_streaming import SpeechTokenBuffer, create_speech_unit_queue, idle_flush_loop


def test_create_speech_unit_queue_bounded():
    q = create_speech_unit_queue()
    if VOICE_SPEECH_QUEUE_MAXSIZE > 0:
        assert q.maxsize == VOICE_SPEECH_QUEUE_MAXSIZE
    else:
        assert q.maxsize == 0


def test_speech_token_buffer_counts_tokens():
    import asyncio

    buf = SpeechTokenBuffer()

    async def run():
        await buf.append_token("hello ")
        await buf.append_token("world")

    asyncio.run(run())
    assert buf.tokens_streamed == 2
    assert "hello world" in buf.buf


def test_is_flushable_fragment_incomplete_clause_is_not_flushable():
    """Regression: this exact fragment used to get force-flushed as its own
    TTS clip after a normal LLM token-batch pause, producing an audible
    mid-sentence gap ("...the thing about {pause} that map.")."""
    assert is_flushable_fragment("Okay, so here's the thing about", chunks_emitted=0) is False


def test_is_flushable_fragment_long_but_mid_clause_is_not_flushable():
    """Regression #2: length alone isn't a safe signal — this fragment is
    11 words / 49 chars (past the first-unit minimum) but is a subordinate
    clause ("when we say X") with no main clause yet, so it still sounds
    broken off ("...when we say "the map" {pause} in this chapter...")."""
    fragment = 'Okay, so here\'s the thing — when we say "the map"'
    assert is_flushable_fragment(fragment, chunks_emitted=0) is False


def test_is_flushable_fragment_complete_short_sentence_is_flushable():
    assert is_flushable_fragment("Okay, so.", chunks_emitted=0) is True


def test_is_flushable_fragment_ending_on_comma_is_flushable():
    """A natural clause pause (comma/semicolon/colon/dash) at the very end
    is where a human speaker would pause too — safe to stop there."""
    assert is_flushable_fragment("Okay, so here's the thing,", chunks_emitted=0) is True


def test_is_flushable_fragment_falls_back_to_max_words_when_no_punctuation():
    """Once a fragment has grown as long as extract_voice_chunks itself
    would force a word-boundary split (max_words), further waiting risks a
    worse stall than just speaking what we have."""
    from app.config import VOICE_SPEECH_MAX_WORDS

    at_max_words = " ".join(["word"] * VOICE_SPEECH_MAX_WORDS)
    assert is_flushable_fragment(at_max_words, chunks_emitted=1) is True
    under_max_words = " ".join(["word"] * (VOICE_SPEECH_MAX_WORDS - 5))
    assert is_flushable_fragment(under_max_words, chunks_emitted=1) is False


def test_idle_flush_loop_waits_longer_for_incomplete_fragment():
    """The core regression test: a mid-clause fragment must not be forced
    out at the normal idle threshold — only once it's had real extra grace
    time (the hard-cap multiplier) with no continuation arriving."""

    async def run():
        buffer = SpeechTokenBuffer()
        buffer.buf = "Okay, so here's the thing about"
        # Backdate past the normal 0.12s first-unit threshold but well short
        # of the 4x hard cap (0.48s).
        buffer.last_token_at = time.monotonic() - 0.2
        queue = create_speech_unit_queue()
        timing = VoicePipelineTiming()
        stop = asyncio.Event()

        task = asyncio.create_task(idle_flush_loop(buffer, queue, stop, timing))
        await asyncio.sleep(0.12)
        stop.set()
        await task
        return queue

    queue = asyncio.run(run())
    assert queue.empty()


def test_idle_flush_loop_eventually_flushes_a_truly_stuck_fragment():
    """A genuinely stuck LLM stream must still be spoken, not silenced
    forever — the hard cap is a last resort, not a removal of the flush."""

    async def run():
        buffer = SpeechTokenBuffer()
        buffer.buf = "Okay, so here's the thing about"
        buffer.last_token_at = time.monotonic() - 0.6  # past the 0.48s hard cap
        queue = create_speech_unit_queue()
        timing = VoicePipelineTiming()
        stop = asyncio.Event()

        task = asyncio.create_task(idle_flush_loop(buffer, queue, stop, timing))
        await asyncio.sleep(0.12)
        stop.set()
        await task
        return queue

    queue = asyncio.run(run())
    assert queue.get_nowait() == "Okay, so here's the thing about"

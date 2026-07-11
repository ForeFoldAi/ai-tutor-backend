"""Phase 2 — voice streaming queue and buffer."""

import asyncio

from app.config import VOICE_SPEECH_QUEUE_MAXSIZE
from app.services.voice_streaming import SpeechTokenBuffer, create_speech_unit_queue


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

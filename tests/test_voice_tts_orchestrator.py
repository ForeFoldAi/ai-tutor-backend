"""Phase 3 — TTS orchestrator peek/clear helpers."""

import asyncio

from app.services.voice_tts_orchestrator import _TTS_STOP, _try_peek, clear_speech_queue


def test_try_peek_returns_next_unit():
    q: asyncio.Queue[str | None] = asyncio.Queue()
    q.put_nowait("hello world")
    assert _try_peek(q) == "hello world"
    assert q.empty()


def test_try_peek_preserves_stop_sentinel():
    q: asyncio.Queue[str | None] = asyncio.Queue()
    q.put_nowait(_TTS_STOP)
    assert _try_peek(q) is None
    assert q.get_nowait() is _TTS_STOP


async def _drain():
    q: asyncio.Queue[str | None] = asyncio.Queue()
    q.put_nowait("a")
    q.put_nowait("b")
    await clear_speech_queue(q)
    assert q.empty()


def test_clear_speech_queue_drains():
    asyncio.run(_drain())

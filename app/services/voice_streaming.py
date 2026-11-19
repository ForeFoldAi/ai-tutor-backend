"""
Phase 2 — voice streaming: speech-unit queue, backpressure, metrics hooks.

Decouples token buffering from TTS so the LLM producer can stream tokens while
speech units wait on a bounded queue (natural backpressure when TTS falls behind).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

from app.config import VOICE_SPEECH_QUEUE_MAXSIZE
from app.services.tts_sanitize import sanitize_chunk_for_tts
from app.services.voice_chunking import (
    IDLE_FLUSH_HARD_MULTIPLIER,
    extract_voice_chunks,
    has_unclosed_math_delimiters,
    idle_flush_sec,
    is_flushable_fragment,
)

if TYPE_CHECKING:
    from app.services.voice_chunking import VoicePipelineTiming


MetricsEmit = Callable[[str], Awaitable[None]]


def create_speech_unit_queue() -> asyncio.Queue[str | None]:
    """
    Bounded queue between LLM token producer and TTS consumer.

    When full, the producer awaits put() — slowing token reads from Mistral
  instead of buffering unbounded text in memory.
    """
    maxsize = VOICE_SPEECH_QUEUE_MAXSIZE
    return asyncio.Queue(maxsize=maxsize if maxsize > 0 else 0)


@dataclass
class SpeechTokenBuffer:
    """Accumulates streamed LLM tokens until extract_voice_chunks can flush units."""

    buf: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_token_at: float = field(default_factory=time.monotonic)
    chunks_emitted: int = 0
    tokens_streamed: int = 0
    # Once a ``` fence opens, everything after it is the math-lesson /
    # science-experiment JSON block the prompt mandates be appended AFTER the
    # prose (see chat_service.py) — never meant to be spoken. Once set, further
    # tokens are dropped from the speech buffer so raw JSON can't reach TTS.
    _fence_open: bool = False

    async def append_token(self, token: str) -> None:
        async with self.lock:
            self.last_token_at = time.monotonic()
            self.tokens_streamed += 1
            if self._fence_open:
                return
            self.buf += token
            idx = self.buf.find("```")
            if idx != -1:
                self.buf = self.buf[:idx]
                self._fence_open = True


async def enqueue_speech_unit(
    queue: asyncio.Queue[str | None],
    timing: VoicePipelineTiming,
    raw_chunk: str,
    buffer: SpeechTokenBuffer | None = None,
    *,
    emit_metrics: MetricsEmit | None = None,
) -> None:
    """Sanitize, enqueue one speech unit, track queue depth for observability."""
    cleaned = sanitize_chunk_for_tts(raw_chunk)
    if not cleaned:
        return

    first_unit = timing.sentences_queued == 0
    timing.mark_sentence_queued(cleaned)
    if buffer is not None:
        buffer.chunks_emitted += 1

    # Backpressure: blocks when TTS queue is full (ponytail: bounded maxsize only)
    await queue.put(cleaned)
    timing.record_queue_depth(queue.qsize())

    if first_unit and emit_metrics is not None:
        await emit_metrics("first_speech_unit")


async def flush_units_from_buffer(
    buffer: SpeechTokenBuffer,
    queue: asyncio.Queue[str | None],
    timing: VoicePipelineTiming,
    *,
    emit_metrics: MetricsEmit | None = None,
) -> None:
    """Extract all ready speech units from the token buffer and enqueue them."""
    async with buffer.lock:
        chunks, buffer.buf = extract_voice_chunks(buffer.buf, chunks_emitted=buffer.chunks_emitted)
    for chunk in chunks:
        await enqueue_speech_unit(queue, timing, chunk, buffer, emit_metrics=emit_metrics)


async def idle_flush_loop(
    buffer: SpeechTokenBuffer,
    queue: asyncio.Queue[str | None],
    stop_event: asyncio.Event,
    timing: VoicePipelineTiming,
    *,
    emit_metrics: MetricsEmit | None = None,
) -> None:
    """
    Flush partial buffer after brief token inactivity.

    Adaptive idle: shorter before first unit (faster first audio),
    slightly longer once speech is flowing (fewer mid-phrase cuts).
    """
    while not stop_event.is_set():
        await asyncio.sleep(0.04)
        async with buffer.lock:
            chunk = buffer.buf.strip()
            idle = time.monotonic() - buffer.last_token_at
            flush_after = idle_flush_sec(chunks_emitted=buffer.chunks_emitted)
            if not chunk or idle < flush_after:
                continue
            if has_unclosed_math_delimiters(chunk):
                continue
            # A short, clause-incomplete fragment ("...the thing about") is
            # ordinary LLM token-batch jitter, not a deliberate pause. Forcing
            # it out now means it gets synthesized as its own separate TTS
            # clip — audibly cut off from what follows. Give fragments like
            # that more idle time before giving up and speaking them as-is.
            if not is_flushable_fragment(chunk, chunks_emitted=buffer.chunks_emitted):
                if idle < flush_after * IDLE_FLUSH_HARD_MULTIPLIER:
                    continue
            buffer.buf = ""

        await enqueue_speech_unit(queue, timing, chunk, buffer, emit_metrics=emit_metrics)

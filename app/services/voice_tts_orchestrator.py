"""
TTS orchestration: multi-slot prefetch, gap metrics, speech-unit sync events.

While unit N plays, units N+1..N+depth synthesize in parallel for warm queues.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import WebSocket

from app.config import VOICE_TTS_PREFETCH, VOICE_TTS_PREFETCH_DEPTH
from app.services import voice_protection_metrics as metrics
from app.services.edge_tts_service import (
    resolve_voice,
    send_mp3_bytes,
    stream_edge_tts,
    synthesize_mp3,
)
from app.services.voice_chunking import VoicePipelineTiming

logger = logging.getLogger(__name__)

_TTS_STOP = None
_QUEUE_POLL_SEC = 0.05
_PREFETCH_SPIN_SEC = 0.012

OnSpeaking = Callable[[], Awaitable[None]]


@dataclass
class _PrefetchSlot:
    text: str
    task: asyncio.Task[bytes]
    needs_task_done: bool = False
    armed_at: float = 0.0

    @classmethod
    def start(
        cls,
        text: str,
        *,
        voice: str,
        stop_event: asyncio.Event,
    ) -> _PrefetchSlot:
        async def _run() -> bytes:
            return await synthesize_mp3(text, voice=voice, stop_event=stop_event)

        return cls(
            text=text,
            task=asyncio.create_task(_run(), name="voice-tts-prefetch"),
            armed_at=time.perf_counter(),
        )

    def done(self) -> bool:
        return self.task.done()

    async def result(self) -> bytes:
        try:
            return self.task.result()
        except Exception as exc:
            logger.debug("Prefetch failed for %r: %s", self.text[:32], exc)
            return b""

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()


async def clear_speech_queue(queue: asyncio.Queue[str | None]) -> None:
    while True:
        try:
            item = queue.get_nowait()
            queue.task_done()
            if item is _TTS_STOP:
                break
        except asyncio.QueueEmpty:
            break


def _try_peek(queue: asyncio.Queue[str | None]) -> str | None:
    try:
        item = queue.get_nowait()
    except asyncio.QueueEmpty:
        return None
    if item is _TTS_STOP:
        queue.put_nowait(_TTS_STOP)
        return None
    if isinstance(item, str) and item.strip():
        return item
    queue.task_done()
    return None


async def _emit_speech_unit(ws: WebSocket, text: str, index: int) -> None:
    """Tell client which chunk is currently spoken (text/speech sync)."""
    try:
        await ws.send_json(
            {
                "type": "speech_unit",
                "text": text,
                "index": index,
                "chars": len(text),
            }
        )
    except Exception:
        pass


async def run_tts_orchestrator(
    websocket: WebSocket,
    speech_queue: asyncio.Queue[str | None],
    stop_event: asyncio.Event,
    timing: VoicePipelineTiming,
    on_speaking: OnSpeaking,
    *,
    voice: str | None = None,
) -> None:
    """
    Consume speech units and emit MP3.

    Prefetch depth keeps the queue warm; bounded speech_queue prevents starvation
    of the LLM producer via backpressure.
    """
    voice = voice or await resolve_voice()
    depth = max(1, VOICE_TTS_PREFETCH_DEPTH) if VOICE_TTS_PREFETCH else 0
    pending: deque[_PrefetchSlot] = deque()
    unit_index = 0
    last_play_end = time.perf_counter()

    def _arm_prefetch_slots() -> None:
        if depth <= 0 or stop_event.is_set():
            return
        while len(pending) < depth and not stop_event.is_set():
            nxt = _try_peek(speech_queue)
            if not nxt:
                break
            slot = _PrefetchSlot.start(nxt, voice=voice, stop_event=stop_event)
            slot.needs_task_done = True
            pending.append(slot)

    async def _play_bytes(text: str, data: bytes, *, from_prefetch: bool) -> None:
        nonlocal unit_index, last_play_end
        if stop_event.is_set():
            return
        now = time.perf_counter()
        gap_ms = (now - last_play_end) * 1000.0
        if unit_index > 0:
            metrics.record_playback_gap(gap_ms)
            timing.mark_playback_gap(gap_ms)

        metrics.set_gauge("tts_queue_depth", float(speech_queue.qsize() + len(pending)))
        await _emit_speech_unit(websocket, text, unit_index)
        await on_speaking()
        timing.mark_tts_unit_played()
        if from_prefetch:
            timing.mark_prefetch_hit()
        else:
            timing.mark_prefetch_miss()

        if not data:
            await stream_edge_tts(
                text, websocket, stop_event, voice=voice, timing=timing, chunk_index=unit_index
            )
        else:
            await send_mp3_bytes(data, websocket, stop_event, timing=timing)

        unit_index += 1
        last_play_end = time.perf_counter()
        if timing.first_chunk_sent_at and unit_index == 1:
            first_audio_ms = (timing.first_chunk_sent_at - timing._t0) * 1000.0
            metrics.set_gauge("first_audio_latency_ms", first_audio_ms)
            metrics.set_gauge("first_tts_latency_ms", first_audio_ms)

    async def _play_slot(slot: _PrefetchSlot) -> None:
        wait_ms = (time.perf_counter() - slot.armed_at) * 1000.0
        metrics.set_gauge("tts_wait_time_ms", wait_ms)
        data = await slot.result()
        if slot.needs_task_done:
            speech_queue.task_done()
        if stop_event.is_set():
            return
        await _play_bytes(slot.text, data, from_prefetch=True)

    try:
        while not stop_event.is_set():
            _arm_prefetch_slots()
            metrics.set_gauge("tts_queue_depth", float(speech_queue.qsize() + len(pending)))

            # Prefer completed prefetch head (gapless)
            if pending and pending[0].done():
                slot = pending.popleft()
                await _play_slot(slot)
                _arm_prefetch_slots()
                continue

            if pending and not pending[0].done():
                await asyncio.sleep(_PREFETCH_SPIN_SEC)
                continue

            # No prefetch ready — pull live text
            try:
                text = await asyncio.wait_for(speech_queue.get(), timeout=_QUEUE_POLL_SEC)
            except asyncio.TimeoutError:
                continue

            if text is _TTS_STOP:
                speech_queue.task_done()
                break
            if not text:
                speech_queue.task_done()
                continue

            _arm_prefetch_slots()
            t_gen = time.perf_counter()
            data = await synthesize_mp3(text, voice=voice, stop_event=stop_event)
            metrics.set_gauge(
                "chunk_generation_latency_ms",
                (time.perf_counter() - t_gen) * 1000.0,
            )
            metrics.set_gauge("chunk_size", float(len(text)))
            speech_queue.task_done()
            await _play_bytes(text, data, from_prefetch=False)

    finally:
        while pending:
            slot = pending.popleft()
            slot.cancel()
            try:
                await slot.task
            except (asyncio.CancelledError, Exception):
                pass
            if slot.needs_task_done:
                try:
                    speech_queue.task_done()
                except ValueError:
                    pass


# Keep clear_speech_queue export name used by voice_ws
tts_worker = run_tts_orchestrator

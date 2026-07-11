"""Phase 6 — HTTP voice fallback TTS with N+1 prefetch between speech units."""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator

from app.config import VOICE_HTTP_TTS_PREFETCH
from app.services.edge_tts_service import resolve_voice, synthesize_mp3

_FRAME_AUDIO = 2


def _frame_audio(data: bytes) -> bytes:
    return struct.pack("<BI", _FRAME_AUDIO, len(data)) + data


async def _mp3_to_frames(data: bytes) -> AsyncIterator[bytes]:
    if not data:
        return
    step = 4096
    for i in range(0, len(data), step):
        yield _frame_audio(data[i : i + step])


class _HttpTtsPrefetcher:
    """Synthesize speech unit N+1 while framing unit N for HTTP clients."""

    def __init__(self) -> None:
        self._voice: str | None = None
        self._next_task: asyncio.Task[bytes] | None = None
        self._next_text: str | None = None

    async def _resolve_voice(self) -> str:
        if self._voice is None:
            self._voice = await resolve_voice()
        return self._voice

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        cleaned = (text or "").strip()
        if not cleaned:
            return b""
        voice_name = voice or await self._resolve_voice()
        if (
            VOICE_HTTP_TTS_PREFETCH
            and self._next_task
            and self._next_text == cleaned
        ):
            try:
                data = await self._next_task
            except Exception:
                data = b""
            self._next_task = None
            self._next_text = None
            if data:
                return data
        if self._next_task and not self._next_task.done():
            self._next_task.cancel()
        self._next_task = None
        self._next_text = None
        return await synthesize_mp3(cleaned, voice=voice_name)

    async def stream_units(
        self, units: list[str], *, voice: str | None = None
    ) -> AsyncIterator[bytes]:
        voice_name = voice or await self._resolve_voice()
        for idx, raw in enumerate(units):
            cleaned = (raw or "").strip()
            if not cleaned:
                continue
            if VOICE_HTTP_TTS_PREFETCH and idx + 1 < len(units):
                nxt = (units[idx + 1] or "").strip()
                if nxt:
                    self._next_text = nxt
                    self._next_task = asyncio.create_task(
                        synthesize_mp3(nxt, voice=voice_name),
                        name="voice-http-tts-prefetch",
                    )
            data = await self.synthesize(cleaned, voice=voice_name)
            async for framed in _mp3_to_frames(data):
                yield framed


_prefetcher: _HttpTtsPrefetcher | None = None


def http_tts_prefetcher() -> _HttpTtsPrefetcher:
    global _prefetcher
    if _prefetcher is None:
        _prefetcher = _HttpTtsPrefetcher()
    return _prefetcher

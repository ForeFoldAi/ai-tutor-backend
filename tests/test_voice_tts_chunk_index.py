"""Prefetch path must pass chunk_index into _communicate."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch


def test_synthesize_mp3_forwards_chunk_index():
    from app.services import edge_tts_service as ets

    seen: list[int] = []

    class FakeCommunicate:
        def stream(self):
            async def _gen():
                return
                yield  # pragma: no cover

            return _gen()

    def _communicate(text, *, voice, chunk_index=0):
        seen.append(chunk_index)
        return FakeCommunicate()

    async def _run() -> None:
        with (
            patch.object(ets, "resolve_voice", new=AsyncMock(return_value="en-IN-NeerjaNeural")),
            patch.object(ets, "_communicate", side_effect=_communicate),
        ):
            await ets.synthesize_mp3("A square has equal sides.", chunk_index=2)

    asyncio.run(_run())
    assert seen == [2]

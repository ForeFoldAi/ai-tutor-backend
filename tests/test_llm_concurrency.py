"""Answers must stream concurrently (the old global lock queued every student behind the previous answer)."""

import asyncio
import time
from unittest.mock import patch

import httpx

from app.services import llm_client

SSE = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'


async def _slow_llm(request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(0.3)
    if b'"stream": true' in request.content or b'"stream":true' in request.content:
        return httpx.Response(200, content=SSE)
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


async def _one_stream() -> str:
    return "".join([t async for t in llm_client.stream([{"role": "user", "content": "q"}])])


def _run(coro):
    with patch.object(llm_client, "_TRANSPORT", httpx.MockTransport(_slow_llm)), \
         patch.object(llm_client, "_LLM_MIN_INTERVAL_SEC", 0.0), \
         patch.object(llm_client, "llm_api_key_for", return_value="test"):
        return asyncio.run(coro)


def test_streams_overlap_instead_of_queueing():
    async def main():
        t = time.perf_counter()
        out = await asyncio.gather(*(_one_stream() for _ in range(3)))
        return out, time.perf_counter() - t

    out, elapsed = _run(main())
    assert out == ["hi", "hi", "hi"]
    assert elapsed < 0.6, f"3 streams took {elapsed:.2f}s; they ran one after another"


def test_slots_released_after_stream_and_complete():
    async def main():
        await _one_stream()
        await llm_client.complete([{"role": "user", "content": "q"}], feature="image_select")
        return llm_client.llm_slot_free("chat"), llm_client.llm_slot_free("image_select")

    assert _run(main()) == (True, True)


def test_retry_sleep_does_not_hold_slot():
    calls = {"n": 0}

    async def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.3"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def main():
        task = asyncio.create_task(llm_client.complete([{"role": "user", "content": "q"}]))
        await asyncio.sleep(0.1)  # first attempt got 429 and is now backing off
        free_during_backoff = llm_client.llm_slot_free("chat")
        return await task, free_during_backoff

    with patch.object(llm_client, "_TRANSPORT", httpx.MockTransport(flaky)), \
         patch.object(llm_client, "_LLM_MIN_INTERVAL_SEC", 0.0), \
         patch.object(llm_client, "_LLM_MAX_CONCURRENCY", 1), \
         patch.object(llm_client, "llm_api_key_for", return_value="test"):
        text, free = asyncio.run(main())
    assert text == "ok"
    assert free, "slot was held while sleeping for Retry-After"

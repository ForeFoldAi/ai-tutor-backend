"""Mistral / LLM client wrappers used by chat."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.student_messages import ANSWER_NOT_IN_CHAPTER
from app.services import llm_client


def _ensure_mistral_config() -> None:
    llm_client.ensure_llm_config()


async def _call_mistral_async(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    feature: str = "chat",
) -> str:
    """Non-blocking OpenAI-compatible chat completion (legacy name kept)."""
    return await llm_client.complete(
        messages,
        feature=feature,
        max_tokens=max_tokens,
        empty_fallback=ANSWER_NOT_IN_CHAPTER,
    )


async def _stream_mistral_async(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    feature: str = "chat",
    temperature: float | None = None,
) -> AsyncIterator[str]:
    """Yield tokens from OpenAI-compatible SSE stream (legacy name kept)."""
    async for token in llm_client.stream(
        messages, feature=feature, max_tokens=max_tokens, temperature=temperature
    ):
        yield token


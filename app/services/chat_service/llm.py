"""Mistral / LLM client wrappers used by chat."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from app.core.student_messages import ANSWER_NOT_IN_CHAPTER
from app.services import llm_client

logger = logging.getLogger(__name__)

_STREAM_ATTEMPTS = int(os.environ.get("LLM_STREAM_ATTEMPTS", "2"))
_STREAM_RETRY_DELAY_SEC = float(os.environ.get("LLM_STREAM_RETRY_DELAY_SEC", "0.6"))


def _ensure_mistral_config() -> None:
    llm_client.ensure_llm_config()


async def _call_mistral_async(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    feature: str = "chat",
    empty_fallback: str = ANSWER_NOT_IN_CHAPTER,
) -> str:
    """Non-blocking OpenAI-compatible chat completion (legacy name kept).

    An empty completion is retried once before falling back — otherwise a transient
    provider hiccup reaches the student as "I couldn't find this in your chapter".
    """
    attempts = max(1, _STREAM_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            text = await llm_client.complete(
                messages, feature=feature, max_tokens=max_tokens, empty_fallback=""
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            if attempt >= attempts:
                raise
            logger.warning(
                "LLM completion failed (attempt %s/%s, feature=%s): %s: %s",
                attempt, attempts, feature, type(exc).__name__, exc,
            )
            await asyncio.sleep(_STREAM_RETRY_DELAY_SEC * attempt)
            continue
        if (text or "").strip():
            return text
        if attempt < attempts:
            logger.warning("LLM returned empty completion (attempt %s/%s); retrying", attempt, attempts)
            await asyncio.sleep(_STREAM_RETRY_DELAY_SEC * attempt)
    return empty_fallback


async def _stream_mistral_async(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    feature: str = "chat",
    temperature: float | None = None,
) -> AsyncIterator[str]:
    """Yield tokens from OpenAI-compatible SSE stream (legacy name kept).

    An empty stream or an error before the first token is retried once — those are
    transient (rate limit, key rotation, provider hiccup) and otherwise surface to the
    student as "I lost my train of thought" (voice) or a raw textbook paste (chat).
    Once any token has been yielded the stream is never restarted (no duplicate text).
    """
    attempts = max(1, _STREAM_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        produced = False
        try:
            async for token in llm_client.stream(
                messages, feature=feature, max_tokens=max_tokens, temperature=temperature
            ):
                if token:
                    produced = True
                yield token
        except Exception as exc:
            if produced or attempt >= attempts:
                raise
            logger.warning(
                "LLM stream failed before first token (attempt %s/%s, feature=%s): %s: %s",
                attempt, attempts, feature, type(exc).__name__, exc,
            )
            await asyncio.sleep(_STREAM_RETRY_DELAY_SEC * attempt)
            continue
        if produced or attempt >= attempts:
            return
        logger.warning(
            "LLM stream returned no content (attempt %s/%s, feature=%s); retrying",
            attempt, attempts, feature,
        )
        await asyncio.sleep(_STREAM_RETRY_DELAY_SEC * attempt)


"""OpenAI-compatible chat completions client (Mistral, Groq, OpenAI, vLLM, …)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    llm_api_key_for,
    llm_base_url_for,
    llm_model_for,
)

logger = logging.getLogger(__name__)


def _chat_url(feature: str) -> str:
    return f"{llm_base_url_for(feature)}/chat/completions"


def _headers(feature: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {llm_api_key_for(feature)}",
        "Content-Type": "application/json",
    }


def ensure_llm_config(feature: str = "chat") -> None:
    if not llm_api_key_for(feature):
        raise FileNotFoundError(
            "Missing LLM_API_KEY (or MISTRAL_API_KEY / per-feature key) env var. "
            "Set it to enable AI answers."
        )


def _payload(
    messages: list[dict[str, str]],
    *,
    feature: str,
    max_tokens: int | None,
    stream: bool = False,
) -> dict[str, Any]:
    token_limit = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    body: dict[str, Any] = {
        "model": llm_model_for(feature),
        "messages": messages,
        "max_tokens": token_limit,
        "temperature": LLM_TEMPERATURE,
    }
    if stream:
        body["stream"] = True
    return body


def _content_from_response(payload: dict[str, Any]) -> str:
    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


async def complete(
    messages: list[dict[str, str]],
    *,
    feature: str = "chat",
    max_tokens: int | None = None,
    empty_fallback: str = "",
) -> str:
    ensure_llm_config(feature)
    body = _payload(messages, feature=feature, max_tokens=max_tokens)
    logger.debug(
        "LLM request feature=%s model=%s base=%s max_tokens=%s",
        feature,
        body["model"],
        llm_base_url_for(feature),
        body["max_tokens"],
    )

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(_chat_url(feature), headers=_headers(feature), json=body)
        resp.raise_for_status()

    text = _content_from_response(resp.json())
    logger.debug("LLM response length=%d chars", len(text))
    return text or empty_fallback


async def stream(
    messages: list[dict[str, str]],
    *,
    feature: str = "chat",
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    ensure_llm_config(feature)
    body = _payload(messages, feature=feature, max_tokens=max_tokens, stream=True)
    logger.debug(
        "LLM streaming request feature=%s model=%s base=%s max_tokens=%s",
        feature,
        body["model"],
        llm_base_url_for(feature),
        body["max_tokens"],
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", _chat_url(feature), headers=_headers(feature), json=body
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


def complete_sync(
    messages: list[dict[str, str]],
    *,
    feature: str = "lesson",
    max_tokens: int | None = None,
    timeout: float = 180.0,
) -> str:
    ensure_llm_config(feature)
    body = _payload(messages, feature=feature, max_tokens=max_tokens)
    logger.debug(
        "LLM sync request feature=%s model=%s base=%s max_tokens=%s",
        feature,
        body["model"],
        llm_base_url_for(feature),
        body["max_tokens"],
    )

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(_chat_url(feature), headers=_headers(feature), json=body)
        resp.raise_for_status()
        return _content_from_response(resp.json())


# ponytail: self-check only; ceiling = import-time assert, upgrade = pytest if suite grows
def _self_check() -> None:
    for feat in ("chat", "voice", "lesson", "assistant"):
        assert llm_model_for(feat) == LLM_MODEL, f"default override leak for {feat}"
        assert llm_base_url_for(feat) == LLM_BASE_URL, f"default base leak for {feat}"
        assert llm_api_key_for(feat) == LLM_API_KEY, f"default key leak for {feat}"


_self_check()

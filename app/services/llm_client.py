"""OpenAI-compatible chat completions client (Mistral, Groq, OpenAI, vLLM, …)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
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

_RETRY_STATUSES = frozenset({429, 503})
# Longer gaps — short retries were burning the same Mistral free-tier budget.
_RETRY_WAITS_SEC = (2.0, 5.0, 15.0)
# Side-car features must not amplify 429 storms (chat answer is primary).
_NO_RETRY_FEATURES = frozenset({"image_select", "voice_affect"})
# Serialize all LLM traffic on this process (chat + image_select share one key).
_LLM_MIN_INTERVAL_SEC = float(os.environ.get("LLM_MIN_INTERVAL_SEC", "0.75"))
_gate = asyncio.Lock()
_sync_gate = __import__("threading").Lock()
_next_ok_at = 0.0
_next_ok_at_sync = 0.0


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
    temperature: float | None = None,
) -> dict[str, Any]:
    token_limit = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    body: dict[str, Any] = {
        "model": llm_model_for(feature),
        "messages": messages,
        "max_tokens": token_limit,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
    }
    if stream:
        body["stream"] = True
    return body


# ponytail: range strip (no `regex` dep); widen blocks if a model invents outliers
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001FAFF"  # pictographs … Extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"  # ZWJ
    "]+"
)


def strip_emojis(text: str) -> str:
    """Remove emoji / pictographs from LLM text so they never reach the UI."""
    if not text:
        return text
    out = _EMOJI_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", out)


def _content_from_response(payload: dict[str, Any]) -> str:
    return strip_emojis(
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    ).strip()


def _max_attempts(feature: str) -> int:
    if feature in _NO_RETRY_FEATURES:
        return 1
    return len(_RETRY_WAITS_SEC) + 1


def _retry_wait_sec(resp: httpx.Response, attempt: int) -> float:
    ra = (resp.headers.get("Retry-After") or "").strip()
    if ra.isdigit():
        return float(ra)
    try:
        return float(ra)
    except ValueError:
        pass
    base = _RETRY_WAITS_SEC[min(attempt, len(_RETRY_WAITS_SEC) - 1)]
    return base + random.uniform(0.0, 0.5)


async def _pace_async() -> None:
    global _next_ok_at
    delay = _next_ok_at - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


def _mark_async_ok(*, cooldown: float = 0.0) -> None:
    global _next_ok_at
    _next_ok_at = time.monotonic() + max(_LLM_MIN_INTERVAL_SEC, cooldown)


def _pace_sync() -> None:
    global _next_ok_at_sync
    delay = _next_ok_at_sync - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _mark_sync_ok(*, cooldown: float = 0.0) -> None:
    global _next_ok_at_sync
    _next_ok_at_sync = time.monotonic() + max(_LLM_MIN_INTERVAL_SEC, cooldown)


async def complete(
    messages: list[dict[str, str]],
    *,
    feature: str = "chat",
    max_tokens: int | None = None,
    empty_fallback: str = "",
    temperature: float | None = None,
) -> str:
    ensure_llm_config(feature)
    body = _payload(messages, feature=feature, max_tokens=max_tokens, temperature=temperature)
    logger.debug(
        "LLM request feature=%s model=%s base=%s max_tokens=%s",
        feature,
        body["model"],
        llm_base_url_for(feature),
        body["max_tokens"],
    )

    async with _gate:
        await _pace_async()
        async with httpx.AsyncClient(timeout=90.0) as client:
            last: httpx.Response | None = None
            attempts = _max_attempts(feature)
            for attempt in range(attempts):
                resp = await client.post(
                    _chat_url(feature), headers=_headers(feature), json=body
                )
                last = resp
                if resp.status_code not in _RETRY_STATUSES:
                    resp.raise_for_status()
                    _mark_async_ok()
                    text = _content_from_response(resp.json())
                    logger.debug("LLM response length=%d chars", len(text))
                    return text or empty_fallback
                wait = _retry_wait_sec(resp, attempt)
                if attempt + 1 >= attempts:
                    _mark_async_ok(cooldown=wait)
                    break
                logger.warning(
                    "LLM complete feature=%s status=%s attempt=%d retry_in=%.1fs",
                    feature,
                    resp.status_code,
                    attempt + 1,
                    wait,
                )
                await asyncio.sleep(wait)
            assert last is not None
            last.raise_for_status()
            return empty_fallback


async def stream(
    messages: list[dict[str, str]],
    *,
    feature: str = "chat",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[str]:
    ensure_llm_config(feature)
    body = _payload(
        messages, feature=feature, max_tokens=max_tokens, stream=True, temperature=temperature
    )
    logger.debug(
        "LLM streaming request feature=%s model=%s base=%s max_tokens=%s",
        feature,
        body["model"],
        llm_base_url_for(feature),
        body["max_tokens"],
    )

    await _gate.acquire()
    try:
        await _pace_async()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response: httpx.Response | None = None
            attempts = _max_attempts(feature)
            for attempt in range(attempts):
                req = client.build_request(
                    "POST", _chat_url(feature), headers=_headers(feature), json=body
                )
                response = await client.send(req, stream=True)
                if response.status_code not in _RETRY_STATUSES:
                    break
                wait = _retry_wait_sec(response, attempt)
                if attempt + 1 >= attempts:
                    _mark_async_ok(cooldown=wait)
                    break
                logger.warning(
                    "LLM stream feature=%s status=%s attempt=%d retry_in=%.1fs",
                    feature,
                    response.status_code,
                    attempt + 1,
                    wait,
                )
                await response.aclose()
                response = None
                await asyncio.sleep(wait)

            assert response is not None
            try:
                response.raise_for_status()
                _mark_async_ok()
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
                            cleaned = strip_emojis(delta)
                            if cleaned:
                                yield cleaned
                    except Exception:
                        continue
            finally:
                await response.aclose()
    finally:
        _gate.release()


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

    with _sync_gate:
        _pace_sync()
        with httpx.Client(timeout=timeout) as client:
            last: httpx.Response | None = None
            attempts = _max_attempts(feature)
            for attempt in range(attempts):
                resp = client.post(_chat_url(feature), headers=_headers(feature), json=body)
                last = resp
                if resp.status_code not in _RETRY_STATUSES:
                    resp.raise_for_status()
                    _mark_sync_ok()
                    return _content_from_response(resp.json())
                wait = _retry_wait_sec(resp, attempt)
                if attempt + 1 >= attempts:
                    _mark_sync_ok(cooldown=wait)
                    break
                logger.warning(
                    "LLM sync feature=%s status=%s attempt=%d retry_in=%.1fs",
                    feature,
                    resp.status_code,
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
            assert last is not None
            last.raise_for_status()
            return ""


# ponytail: self-check only; ceiling = import-time assert, upgrade = pytest if suite grows
def _self_check() -> None:
    for feat in ("chat", "voice", "lesson", "assistant"):
        assert llm_model_for(feat) == LLM_MODEL, f"default override leak for {feat}"
        assert llm_base_url_for(feat) == LLM_BASE_URL, f"default base leak for {feat}"
        assert llm_api_key_for(feat) == LLM_API_KEY, f"default key leak for {feat}"
    assert _retry_wait_sec(httpx.Response(429, headers={"Retry-After": "3"}), 0) == 3.0
    assert _max_attempts("image_select") == 1
    assert _max_attempts("chat") == len(_RETRY_WAITS_SEC) + 1
    assert "Concept" in strip_emojis("🌱 Concept 💡 tip ✅")
    assert "🌱" not in strip_emojis("🌱 Concept") and "💡" not in strip_emojis("💡 tip")


_self_check()

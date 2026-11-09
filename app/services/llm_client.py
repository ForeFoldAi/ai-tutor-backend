"""OpenAI-compatible chat completions client (Mistral, Groq, OpenAI, vLLM, …)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_FALLBACK_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    llm_api_key_for,
    llm_key_pool_for,
    llm_base_url_for,
    llm_fallback_for,
    llm_model_for,
    model_extra_params,
)
from app.services import key_pool
from app.services.token_usage import current_user_id, record_usage, usage_from

logger = logging.getLogger(__name__)


def _track(feature: str, model: str, payload: dict[str, Any], *, in_loop: bool) -> None:
    usage = usage_from(payload)
    if not usage:
        return
    args = (current_user_id.get(), feature, model, *usage)
    if in_loop:
        asyncio.get_running_loop().run_in_executor(None, record_usage, *args)
    else:
        record_usage(*args)

_RETRY_STATUSES = frozenset({429, 503})
# Longer gaps — short retries were burning the same Mistral free-tier budget.
_RETRY_WAITS_SEC = (2.0, 5.0, 15.0)
# Side-car features must not amplify 429 storms (chat answer is primary).
# "vision" retries are owned by image_understanding/vision.py (VISION_MAX_RETRIES) to avoid nesting.
_NO_RETRY_FEATURES = frozenset({"image_select", "voice_affect", "voice_understanding", "vision"})
# Concurrent answers per process. Raise on a paid tier; 429s still back everyone off via cooldown.
_LLM_MAX_CONCURRENCY = max(1, int(os.environ.get("LLM_MAX_CONCURRENCY", "8")))
# Side-car calls get their own single slot so they never delay a student's answer.
_SIDE_FEATURES = frozenset({"image_select", "voice_affect", "voice_understanding"})
# Minimum gap between request *starts* (not durations) across the process.
_LLM_MIN_INTERVAL_SEC = float(os.environ.get("LLM_MIN_INTERVAL_SEC", "0.0"))
_pace_lock = threading.Lock()
_next_start_at: dict[str, float] = {}  # per provider base URL
# Streaming: give up on a provider that sends no first token within this window.
_FIRST_TOKEN_TIMEOUT_SEC = float(os.environ.get("LLM_FIRST_TOKEN_TIMEOUT_SEC", "6"))
_sync_gate = threading.BoundedSemaphore(_LLM_MAX_CONCURRENCY)
_sync_client = httpx.Client(timeout=180.0)
# Tests inject an httpx.MockTransport here.
_TRANSPORT: httpx.AsyncBaseTransport | None = None
# asyncio primitives and pooled connections belong to one event loop (lesson planner runs its own).
_loop_state: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[httpx.AsyncClient, asyncio.Semaphore, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


def _async_state() -> tuple[httpx.AsyncClient, asyncio.Semaphore, asyncio.Semaphore]:
    loop = asyncio.get_running_loop()
    state = _loop_state.get(loop)
    if state is None:
        client = httpx.AsyncClient(timeout=120.0, transport=_TRANSPORT)
        state = _loop_state[loop] = (client, asyncio.Semaphore(_LLM_MAX_CONCURRENCY), asyncio.Semaphore(1))
    return state


def _async_slot(feature: str) -> asyncio.Semaphore:
    _client, answers, side = _async_state()
    return side if feature in _SIDE_FEATURES else answers


def llm_slot_free(feature: str = "chat") -> bool:
    """True when a call for ``feature`` would start without queueing (drives the 'waiting' status)."""
    return not _async_slot(feature).locked()


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
    messages: list[dict[str, Any]],
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
    body.update(model_extra_params(body["model"]))
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


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def _content_from_response(payload: dict[str, Any]) -> str:
    return _THINK_RE.sub("", strip_emojis(
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )).strip()


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


# A parked key is tried last; never make a student wait out a long Retry-After for it.
_MAX_PACE_WAIT_SEC = float(os.environ.get("LLM_MAX_PACE_WAIT_SEC", "3"))
# A primary key free again within this many seconds is still tried before the secondary provider.
_PRIMARY_WAIT_SEC = min(_MAX_PACE_WAIT_SEC, float(os.environ.get("LLM_PRIMARY_WAIT_SEC", "2")))
# One key rejected (revoked, no access, request over its token limit): try the next key.
_KEY_FAIL_STATUSES = frozenset({401, 403, 413})
_BAD_KEY_PARK_SEC = 300.0


def _reserve_start(provider: str = "") -> float:
    """Claim the next start time for a provider; returns seconds to wait for it."""
    with _pace_lock:
        now = time.monotonic()
        start = max(now, _next_start_at.get(provider, 0.0))
        _next_start_at[provider] = start + _LLM_MIN_INTERVAL_SEC
        return start - now


def _cool_down(wait: float, provider: str = "") -> None:
    """After a 429/503 nobody starts a new call to *that provider* for ``wait`` seconds."""
    with _pace_lock:
        _next_start_at[provider] = max(_next_start_at.get(provider, 0.0), time.monotonic() + wait)


async def _pace_async(provider: str = "") -> None:
    delay = _reserve_start(provider)
    if delay > 0:
        await asyncio.sleep(min(delay, _MAX_PACE_WAIT_SEC))


def _pace_sync(provider: str = "") -> None:
    delay = _reserve_start(provider)
    if delay > 0:
        time.sleep(delay)


def _key_targets(base_url: str, keys: list[str], model: str) -> list[dict[str, str]]:
    """One target per API key (round-robin start, rate-limited keys last)."""
    return [
        {
            "name": f"{base_url}#{key_pool.label(k)}",
            "scope": base_url,
            "base_url": base_url,
            "key": k,
            "model": model,
        }
        for k in key_pool.ordered(base_url, keys)
    ]


def _targets(feature: str) -> list[dict[str, str]]:
    """Every primary key that is usable now or within a moment, then the secondary's keys."""
    primary = _key_targets(llm_base_url_for(feature), llm_key_pool_for(feature), llm_model_for(feature))
    soon = [t for t in primary if key_pool.seconds_until_ready(t["scope"], t["key"]) <= _PRIMARY_WAIT_SEC]
    parked = [t for t in primary if t not in soon]
    out = soon
    fb = llm_fallback_for(feature)
    if fb and fb["key"]:
        out = out + _key_targets(fb["base_url"], fb.get("keys") or [fb["key"]], fb["model"])
    # Long rate-limited keys are a last resort: the secondary answers before we retry them.
    return out + parked


def _park(target: dict[str, str], wait: float) -> None:
    """Rate-limited/failing key: nobody uses it for ``wait`` s; traffic moves to the next key."""
    _cool_down(wait, target["name"])
    key_pool.cool(target.get("scope", target["base_url"]), target["key"], wait)


def _target_body(body: dict[str, Any], target: dict[str, str]) -> dict[str, Any]:
    out = {k: v for k, v in body.items() if k != "reasoning_effort"}
    return {**out, "model": target["model"], **model_extra_params(target["model"])}


def _target_request(client: httpx.AsyncClient, target: dict[str, str], body: dict[str, Any], *, stream: bool, timeout: float):
    headers = {"Authorization": f"Bearer {target['key']}", "Content-Type": "application/json"}
    url = f"{target['base_url']}/chat/completions"
    payload = _target_body(body, target)
    if stream:
        return client.send(client.build_request("POST", url, headers=headers, json=payload), stream=True)
    return client.post(url, headers=headers, json=payload, timeout=timeout)


async def _send_with_retries(
    feature: str, body: dict[str, Any], *, stream: bool = False, timeout: float = 90.0
) -> httpx.Response:
    """Send under a concurrency slot with instant provider failover.

    Every usable primary key is tried before the fallback provider: a 429/5xx or a
    rejected key (401/403) moves to the next key. A timeout or a request too large
    for the provider (413) is the same for all its keys, so the rest are skipped.
    We only back off once every provider has failed.
    For ``stream=True`` the slot stays held on return; caller must release it.
    """
    client, *_ = _async_state()
    slot = _async_slot(feature)
    targets = _targets(feature)
    attempts = max(_max_attempts(feature), len(targets))
    last_exc: BaseException | None = None
    skip_scopes: set[str] = set()
    for attempt in range(attempts):
        target = targets[attempt % len(targets)]
        scope = target.get("scope", target["base_url"])
        others_left = any(t.get("scope", t["base_url"]) not in skip_scopes for t in targets)
        if scope in skip_scopes and others_left and attempt + 1 < attempts:
            continue
        await slot.acquire()
        try:
            await _pace_async(target["name"])
            # Streaming: bound the wait for response headers/first byte so a stalled provider fails over.
            if stream and len(targets) > 1:
                resp = await asyncio.wait_for(
                    _target_request(client, target, body, stream=True, timeout=timeout),
                    timeout=_FIRST_TOKEN_TIMEOUT_SEC,
                )
            else:
                resp = await _target_request(client, target, body, stream=stream, timeout=timeout)
        except (asyncio.TimeoutError, httpx.TransportError) as exc:
            slot.release()
            last_exc = exc
            logger.warning("LLM feature=%s provider=%s %s — failing over", feature, target["name"], exc.__class__.__name__)
            _park(target, 2.0)
            skip_scopes.add(scope)
            if attempt + 1 >= attempts:
                raise
            continue
        except BaseException:
            slot.release()
            raise
        code = resp.status_code
        failed = code in _RETRY_STATUSES or code in _KEY_FAIL_STATUSES or code >= 500
        if not failed or attempt + 1 >= attempts:
            if failed and code != 413:
                _park(target, _retry_wait_sec(resp, attempt))
            if not stream:
                slot.release()
            return resp
        wait = _retry_wait_sec(resp, attempt)
        if code == 413:
            skip_scopes.add(scope)
        elif code in (401, 403):
            _park(target, _BAD_KEY_PARK_SEC)
        else:
            _park(target, wait)
        if stream:
            await resp.aclose()
        slot.release()
        logger.warning(
            "LLM feature=%s provider=%s status=%s attempt=%d", feature, target["name"], resp.status_code, attempt + 1
        )
        # Another provider is available -> switch immediately; otherwise back off.
        if len(targets) == 1 or (attempt + 1) % len(targets) == 0:
            await asyncio.sleep(min(wait, 5.0))
    if last_exc:
        raise last_exc
    raise AssertionError("unreachable")


async def complete(
    messages: list[dict[str, Any]],
    *,
    feature: str = "chat",
    max_tokens: int | None = None,
    empty_fallback: str = "",
    temperature: float | None = None,
    timeout: float | None = None,
) -> str:
    ensure_llm_config(feature)
    body = _payload(messages, feature=feature, max_tokens=max_tokens, temperature=temperature)
    logger.debug("LLM request feature=%s model=%s max_tokens=%s", feature, body["model"], body["max_tokens"])
    client_timeout = 60.0 if timeout is None else float(timeout)

    resp = await _send_with_retries(feature, body, timeout=client_timeout)
    resp.raise_for_status()
    data = resp.json()
    _track(feature, body["model"], data, in_loop=True)
    text = _content_from_response(data)
    return text or empty_fallback


async def stream(
    messages: list[dict[str, Any]],
    *,
    feature: str = "chat",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[str]:
    ensure_llm_config(feature)
    body = _payload(
        messages, feature=feature, max_tokens=max_tokens, stream=True, temperature=temperature
    )
    logger.debug("LLM streaming request feature=%s model=%s", feature, body["model"])

    response = await _send_with_retries(feature, body, stream=True)
    try:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                _track(feature, body["model"], chunk, in_loop=True)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    cleaned = strip_emojis(delta)
                    if cleaned:
                        yield cleaned
            except Exception:
                continue
    finally:
        try:
            await response.aclose()
        finally:
            _async_slot(feature).release()


def complete_sync(
    messages: list[dict[str, Any]],
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

    attempts = _max_attempts(feature)
    pool = llm_key_pool_for(feature)
    for attempt in range(attempts):
        key = key_pool.ordered(llm_base_url_for(feature), pool)[0] if pool else llm_api_key_for(feature)
        with _sync_gate:
            _pace_sync(llm_base_url_for(feature))
            resp = _sync_client.post(
                _chat_url(feature),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
        if resp.status_code not in _RETRY_STATUSES:
            resp.raise_for_status()
            data = resp.json()
            _track(feature, body["model"], data, in_loop=False)
            return _content_from_response(data)
        wait = _retry_wait_sec(resp, attempt)
        if key:
            key_pool.cool(llm_base_url_for(feature), key, wait)
        other_ready = any(
            not key_pool.is_cooling(llm_base_url_for(feature), k) for k in pool if k != key
        )
        if other_ready:
            wait = 0.0  # another key is free: switch immediately, no pacing cooldown
        else:
            _cool_down(wait, llm_base_url_for(feature))
        if attempt + 1 >= attempts:
            resp.raise_for_status()
        logger.warning(
            "LLM sync feature=%s status=%s attempt=%d retry_in=%.1fs",
            feature,
            resp.status_code,
            attempt + 1,
            wait,
        )
        time.sleep(wait)
    return ""


# ponytail: self-check only; ceiling = import-time assert, upgrade = pytest if suite grows
def _self_check() -> None:
    for feat in ("chat", "voice", "assistant"):
        assert llm_model_for(feat) == LLM_MODEL, f"default override leak for {feat}"
        assert llm_base_url_for(feat) == LLM_BASE_URL, f"default base leak for {feat}"
        assert llm_api_key_for(feat) == LLM_API_KEY, f"default key leak for {feat}"
    # Vision uses a dedicated default — must not silently reuse a text-only chat model.
    assert llm_model_for("vision"), "vision model missing"
    assert _retry_wait_sec(httpx.Response(429, headers={"Retry-After": "3"}), 0) == 3.0
    assert _max_attempts("image_select") == 1
    assert _max_attempts("chat") == len(_RETRY_WAITS_SEC) + 1
    assert "Concept" in strip_emojis("🌱 Concept 💡 tip ✅")
    assert "🌱" not in strip_emojis("🌱 Concept") and "💡" not in strip_emojis("💡 tip")
    # Multimodal payload shape accepted
    body = _payload(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        feature="vision",
        max_tokens=10,
    )
    assert isinstance(body["messages"][0]["content"], list)


_self_check()
logger.info(
    "LLM provider=%s keys=%d model=%s fallback=%s",
    LLM_PROVIDER,
    len(llm_key_pool_for("chat")),
    LLM_MODEL,
    LLM_FALLBACK_PROVIDER or "none",
)

"""Analysis-only cache (never caches tutor answers)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from app.config import IMAGE_ANALYSIS_VERSION, IMAGE_CACHE_ENABLED, IMAGE_CACHE_TTL
from app.modules.image_understanding.schemas import ImageUnderstandingResult, sanitize_vision_payload

if TYPE_CHECKING:
    from app.modules.image_understanding.ocr.schemas import OCRResult

logger = logging.getLogger(__name__)

# ponytail: process-local dict (oldest-first eviction at _MEMORY_MAX); Redis when IMAGE_CACHE_ENABLED and up
_MEMORY: dict[str, tuple[float, dict[str, Any]]] = {}
_MEMORY_MAX = 512
# After a Redis failure, skip it for this long instead of paying the connect timeout on every lookup.
_REDIS_BACKOFF_SEC = 30.0
_redis_down_until = 0.0


def _redis_call(fn):
    global _redis_down_until
    if time.monotonic() < _redis_down_until:
        return None
    try:
        import redis

        from app.core.config import get_settings

        r = redis.from_url(get_settings().redis_url, socket_connect_timeout=0.5, socket_timeout=0.5)
        try:
            return fn(r)
        finally:
            r.close()
    except Exception:
        _redis_down_until = time.monotonic() + _REDIS_BACKOFF_SEC
        logger.debug("image analysis cache: redis unavailable, backing off %.0fs", _REDIS_BACKOFF_SEC)
        return None


def _remember(key: str, payload: dict[str, Any]) -> None:
    _MEMORY.pop(key, None)
    while len(_MEMORY) >= _MEMORY_MAX:
        _MEMORY.pop(next(iter(_MEMORY)))
    _MEMORY[key] = (time.time() + IMAGE_CACHE_TTL, payload)


def _get(key: str) -> dict[str, Any] | None:
    if not IMAGE_CACHE_ENABLED:
        return None
    hit = _MEMORY.get(key)
    if hit:
        expires, payload = hit
        if expires > time.time():
            return payload
        _MEMORY.pop(key, None)
    raw = _redis_call(lambda r: r.get(key))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    _remember(key, payload)
    return payload


def _set(key: str, payload: dict[str, Any]) -> None:
    if not IMAGE_CACHE_ENABLED:
        return
    _remember(key, payload)
    _redis_call(lambda r: r.setex(key, IMAGE_CACHE_TTL, json.dumps(payload)))


def cache_key(content_hash: str, model: str, message: str = "") -> str:
    """The student message steers the vision prompt, so different questions on one image don't share a result."""
    key = f"imgund:{IMAGE_ANALYSIS_VERSION}:{model}:{content_hash}"
    norm = " ".join(message.lower().split())
    return f"{key}:{hashlib.sha256(norm.encode()).hexdigest()[:16]}" if norm else key


def ocr_cache_key(content_hash: str, engine: str, lang: str) -> str:
    return f"imgocr:{IMAGE_ANALYSIS_VERSION}:{engine}:{lang}:{content_hash}"


def get_cached_analysis(content_hash: str, model: str, message: str = "") -> ImageUnderstandingResult | None:
    payload = _get(cache_key(content_hash, model, message))
    return sanitize_vision_payload(payload) if payload else None


def set_cached_analysis(content_hash: str, model: str, result: ImageUnderstandingResult, message: str = "") -> None:
    _set(cache_key(content_hash, model, message), result.model_dump())


def get_cached_ocr(content_hash: str, engine: str, lang: str) -> OCRResult | None:
    from app.modules.image_understanding.ocr.schemas import OCRResult

    payload = _get(ocr_cache_key(content_hash, engine, lang))
    if not payload:
        return None
    try:
        return OCRResult.model_validate(payload)
    except Exception:
        return None


def set_cached_ocr(content_hash: str, engine: str, lang: str, result: OCRResult) -> None:
    # Word boxes only feed text_coverage, which is already computed; don't cache them.
    _set(ocr_cache_key(content_hash, engine, lang), result.model_dump(exclude={"words"}))

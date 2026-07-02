"""
Redis cache helpers for the AI Tutor chat layer.

All functions are async-safe and fail silently — a Redis outage never breaks
the chat endpoint; it simply bypasses the cache.

Cache key: sha256(collection_name + sorted_chapter_ids + normalised_query)
Default TTL: 24 hours (86 400 s)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


def _make_key(
    collection: str,
    chapter_ids: list[str] | None,
    query: str,
    class_level: str = "",
) -> str:
    # v26: include class_level so answers never bleed across grades.
    canonical = (
        f"v26|{collection}|{class_level or ''}|"
        f"{','.join(sorted(chapter_ids or []))}|{query.strip().lower()}"
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"tutor:qa:{digest}"


async def get_cached_answer(
    collection: str,
    chapter_ids: list[str] | None,
    query: str,
    class_level: str = "",
) -> str | None:
    """Return cached answer string, or None on miss / error."""
    try:
        r = get_redis()
        key = _make_key(collection, chapter_ids, query, class_level)
        value = await r.get(key)
        if value:
            logger.debug("Cache HIT for key=%s", key[:16])
        return value
    except Exception as exc:
        logger.warning("Redis GET failed (cache miss): %s", exc)
        return None


def deserialize_tutor_cache(value: str) -> tuple[str, dict[str, Any] | None]:
    """Return (answer text, math_lesson dict) from cache. Images are never stored."""
    raw = (value or "").strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "answer" in obj:
                lesson = obj.get("math_lesson")
                if isinstance(lesson, dict):
                    return str(obj.get("answer") or ""), lesson
                return str(obj.get("answer") or ""), None
        except Exception:
            pass
    return raw, None


def serialize_tutor_cache(
    answer: str,
    related_images: list[dict],
    *,
    math_lesson: dict[str, Any] | None = None,
) -> str:
    # Images are intentionally excluded — they are always re-ranked per request.
    payload: dict[str, Any] = {"answer": answer}
    if math_lesson:
        payload["math_lesson"] = math_lesson
    return json.dumps(payload, ensure_ascii=False)


async def set_cached_answer(
    collection: str,
    chapter_ids: list[str] | None,
    query: str,
    answer: str,
    ttl: int = 86_400,
    *,
    related_images: list[dict] | None = None,
    math_lesson: dict[str, Any] | None = None,
    class_level: str = "",
) -> None:
    """Store answer text (and optional math lesson) in cache. Images are never cached."""
    try:
        r = get_redis()
        key = _make_key(collection, chapter_ids, query, class_level)
        payload = serialize_tutor_cache(answer, [], math_lesson=math_lesson)
        await r.set(key, payload, ex=ttl)
        logger.debug("Cache SET key=%s ttl=%ds", key[:16], ttl)
    except Exception as exc:
        logger.warning("Redis SET failed: %s", exc)


async def invalidate_collection(collection: str) -> None:
    """Delete all cached answers for a collection (call after re-embedding a subject)."""
    try:
        r = get_redis()
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await r.scan(cursor, match=f"tutor:qa:*", count=200)
            # We cannot filter by collection cheaply without storing a reverse index,
            # so for now we clear the entire QA cache when a collection changes.
            # This is safe: cache misses are transparent.
            if keys:
                await r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("Invalidated %d cache entries for collection '%s'", deleted, collection)
    except Exception as exc:
        logger.warning("Cache invalidation failed: %s", exc)

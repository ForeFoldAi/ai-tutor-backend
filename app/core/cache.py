"""
Async Redis client helper (shared).

Used by voice learner profiles (fail-open). Tutor Q&A answers are NOT cached in Redis.
Celery / lesson planner use the sync Redis client in lesson_planner.redis.job_state.
"""

from __future__ import annotations

import logging
from typing import Optional

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

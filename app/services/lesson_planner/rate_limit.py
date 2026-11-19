from __future__ import annotations

import logging
import time

import redis
from fastapi import HTTPException, status

from app.services.lesson_planner.redis.job_state import get_sync_redis

logger = logging.getLogger(__name__)

# ponytail: per-user fixed window — upgrade to sliding window + tiered limits
_WINDOW_SECONDS = 60
_MAX_REQUESTS = 10


def _redis_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Lesson planner requires Redis. Start Redis (e.g. docker compose up redis -d) and the lesson worker.",
    )


def check_rate_limit(user_id: str, client_ip: str) -> None:
    try:
        r = get_sync_redis()
        bucket = int(time.time()) // _WINDOW_SECONDS
        key = f"lesson_planner:rate:{user_id}:{bucket}"
        count = r.incr(key)
        if count == 1:
            r.expire(key, _WINDOW_SECONDS + 5)
        if count > _MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for lesson planner generation",
            )
    except redis.ConnectionError as exc:
        logger.warning("Lesson planner rate limit skipped — Redis unavailable: %s", exc)
        raise _redis_unavailable() from exc

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from app.core.config import get_settings
from app.modules.teacher.lesson_planner.constants import REDIS_CANCEL_PREFIX, REDIS_JOB_PREFIX, REDIS_PUBSUB_CHANNEL

logger = logging.getLogger(__name__)

_sync_redis: redis.Redis | None = None


def get_sync_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is None:
        settings = get_settings()
        _sync_redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _sync_redis


def job_state_key(job_id: str) -> str:
    return f"{REDIS_JOB_PREFIX}{job_id}"


def cancel_key(job_id: str) -> str:
    return f"{REDIS_CANCEL_PREFIX}{job_id}"


def pubsub_channel(job_id: str) -> str:
    return f"{REDIS_PUBSUB_CHANNEL}{job_id}"


def set_job_state(job_id: str, state: dict[str, Any], *, ttl_seconds: int = 86_400) -> None:
    r = get_sync_redis()
    r.set(job_state_key(job_id), json.dumps(state), ex=ttl_seconds)


def get_job_state(job_id: str) -> dict[str, Any] | None:
    r = get_sync_redis()
    raw = r.get(job_state_key(job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def publish_event(job_id: str, event: dict[str, Any]) -> None:
    r = get_sync_redis()
    payload = json.dumps(event, ensure_ascii=False)
    r.publish(pubsub_channel(job_id), payload)
    # ponytail: also merge into job state for late WebSocket subscribers
    state = get_job_state(job_id) or {"events": []}
    events = state.setdefault("events", [])
    events.append(event)
    state["last_event"] = event
    if event.get("event") == "progress":
        state["progress"] = event.get("progress", 0)
        state["message"] = event.get("message")
    set_job_state(job_id, state)


def request_cancel(job_id: str) -> None:
    r = get_sync_redis()
    r.set(cancel_key(job_id), "1", ex=3600)


def is_cancelled(job_id: str) -> bool:
    r = get_sync_redis()
    return bool(r.get(cancel_key(job_id)))


def clear_cancel(job_id: str) -> None:
    r = get_sync_redis()
    r.delete(cancel_key(job_id))

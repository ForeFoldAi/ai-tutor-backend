"""Fail-open probes for DB, Redis, Celery, Qdrant, and Voice."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def probe_database() -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        from sqlalchemy import text

        from app.core.database import SessionLocal

        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return {"status": "healthy", "latency_ms": (time.perf_counter() - t0) * 1000}
        finally:
            db.close()
    except Exception as exc:
        logger.warning("db probe failed: %s", exc)
        return {"status": "down", "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}


def probe_redis() -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        import redis

        from app.core.config import get_settings

        r = redis.from_url(get_settings().redis_url, socket_connect_timeout=2, socket_timeout=2)
        r.ping()
        return {"status": "healthy", "latency_ms": (time.perf_counter() - t0) * 1000}
    except Exception as exc:
        logger.warning("redis probe failed: %s", exc)
        return {"status": "down", "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}


def probe_celery() -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        from app.core.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=0.6)
        pings = inspector.ping() or {}
        active = inspector.active() or {}
        online = len(pings)
        active_tasks = sum(len(v or []) for v in active.values())
        status = "healthy" if online > 0 else "degraded"
        return {
            "status": status,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "workers_online": online,
            "active_tasks": active_tasks,
        }
    except Exception as exc:
        logger.warning("celery probe failed: %s", exc)
        return {"status": "down", "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}


def probe_qdrant() -> dict[str, Any]:
    t0 = time.perf_counter()
    url = (os.environ.get("QDRANT_URL") or "http://127.0.0.1:6333").rstrip("/")
    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(f"{url}/readyz")
            ok = resp.status_code < 400
            if not ok:
                resp = client.get(f"{url}/")
                ok = resp.status_code < 500
        return {
            "status": "healthy" if ok else "degraded",
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    except Exception as exc:
        logger.warning("qdrant probe failed: %s", exc)
        return {"status": "down", "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}


def probe_voice() -> dict[str, Any]:
    t0 = time.perf_counter()
    base = (os.environ.get("VOICE_URL") or os.environ.get("TUTOR_VOICE_URL") or "").rstrip("/")
    if not base:
        # Same-host fallback used in some deploys
        base = "http://127.0.0.1:8080"
    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(f"{base}/rtc/health")
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        status = "healthy"
        if resp.status_code >= 500:
            status = "down"
        elif data.get("status") == "draining" or data.get("redis") == "down":
            status = "degraded"
        return {
            "status": status,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "detail": {
                "liveSessions": data.get("liveSessions"),
                "redis": data.get("redis"),
                "draining": data.get("draining"),
            },
        }
    except Exception as exc:
        logger.warning("voice probe failed: %s", exc)
        return {"status": "down", "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}


def probe_all() -> dict[str, dict[str, Any]]:
    return {
        "postgres": probe_database(),
        "redis": probe_redis(),
        "worker": probe_celery(),
        "qdrant": probe_qdrant(),
        "voice": probe_voice(),
        "api": {"status": "healthy", "latency_ms": 0.0},
    }

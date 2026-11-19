"""Fail-open probes for DB, Redis, Celery, Qdrant, Voice, and Website."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _env_url(*keys: str) -> str:
    for key in keys:
        raw = (os.environ.get(key) or "").strip().rstrip("/")
        if raw:
            return raw
    return ""


def _json_body(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    url = _env_url("QDRANT_URL") or "http://127.0.0.1:6333"
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


def _voice_base_candidates() -> list[str]:
    """Resolve Nest voice base URL. Prefer explicit env; never pretend localhost is prod voice."""
    bases: list[str] = []
    for key in ("VOICE_URL", "TUTOR_VOICE_URL", "VOICE_PUBLIC_URL", "VOICE_INTERNAL_URL"):
        val = _env_url(key)
        if val and val not in bases:
            bases.append(val)
    env = (os.environ.get("ENVIRONMENT") or "").strip().lower()
    # Local/dev only fallback — inside Coolify API container :8080 is not voice.
    if not bases and env in ("", "local", "development", "dev"):
        bases.append("http://127.0.0.1:8080")
    return bases


def probe_voice() -> dict[str, Any]:
    t0 = time.perf_counter()
    bases = _voice_base_candidates()
    if not bases:
        return {
            "status": "unknown",
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": "VOICE_URL not set (expected Nest base, e.g. https://api.voice.tutor.forefoldai.com)",
        }

    last_error = ""
    for base in bases:
        try:
            with httpx.Client(timeout=2.5, follow_redirects=True) as client:
                resp = client.get(f"{base}/rtc/health")
                data = _json_body(resp)
            status = "healthy"
            body_status = str(data.get("status") or "").lower()
            if resp.status_code >= 500:
                status = "down"
            elif body_status in ("draining", "down", "error"):
                status = "degraded" if body_status == "draining" else "down"
            elif data.get("redis") in ("down", "error"):
                status = "degraded"
            elif resp.status_code >= 400 and body_status not in ("ok", "healthy", "up"):
                status = "degraded"
            return {
                "status": status,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "probed_url": f"{base}/rtc/health",
                "detail": {
                    "liveSessions": data.get("liveSessions"),
                    "redis": data.get("redis"),
                    "draining": data.get("draining"),
                    "instanceId": data.get("instanceId"),
                },
            }
        except Exception as exc:
            last_error = str(exc)
            logger.warning("voice probe failed for %s: %s", base, exc)

    return {
        "status": "down",
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "error": last_error or "voice unreachable",
        "tried": bases,
    }


def probe_website() -> dict[str, Any]:
    t0 = time.perf_counter()
    base = _env_url("FRONTEND_URL", "PUBLIC_APP_URL", "APP_URL", "WEBSITE_URL")
    if not base:
        return {
            "status": "unknown",
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": "FRONTEND_URL not set",
        }
    try:
        with httpx.Client(timeout=3.0, follow_redirects=True) as client:
            resp = client.get(base)
        ok = resp.status_code < 500
        return {
            "status": "healthy" if ok else "down",
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "probed_url": base,
            "http_status": resp.status_code,
        }
    except Exception as exc:
        logger.warning("website probe failed: %s", exc)
        return {"status": "down", "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}


def probe_all() -> dict[str, dict[str, Any]]:
    website = probe_website()
    return {
        "postgres": probe_database(),
        "redis": probe_redis(),
        "worker": probe_celery(),
        "qdrant": probe_qdrant(),
        "voice": probe_voice(),
        # MonService key is `frontend` (label: Website)
        "frontend": website,
        "website": website,
        "api": {"status": "healthy", "latency_ms": 0.0},
    }

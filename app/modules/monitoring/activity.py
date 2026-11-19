"""Active-user + per-user LLM route activity for Master Admin monitoring."""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Product surfaces shown in Monitor → AI Tutor
SURFACE_LABELS: dict[str, str] = {
    "ask_learn_mastro": "Ask Learn Mastro",
    "ask_questions": "Ask Questions",
    "practice_problems": "Practice Problems",
    "explain_topic": "Explain a Topic",
    "learn_mastro_text": "Learn Mastro (Text to Text)",
    "ai_voice_tutor": "AI Voice Tutor",
}

ACTIVE_WINDOW_SEC = 10 * 60  # 10 minutes

current_product_surface: ContextVar[str | None] = ContextVar("mon_product_surface", default=None)
current_agent_mode: ContextVar[str | None] = ContextVar("mon_agent_mode", default=None)

_lock = threading.Lock()
# user_id -> activity payload
_user_activity: dict[int, dict[str, Any]] = {}


def bind_request_surface(*, agent_mode: str | None = None, voice_mode: bool = False, feature: str = "chat") -> str:
    """Set context vars for the current request and return the resolved surface key."""
    mode = (agent_mode or "").strip().lower() or None
    current_agent_mode.set(mode)
    surface = resolve_surface(feature=feature, agent_mode=mode, voice_mode=voice_mode)
    current_product_surface.set(surface)
    return surface


def resolve_surface(*, feature: str, agent_mode: str | None = None, voice_mode: bool = False) -> str:
    """Map backend feature/mode to a Monitor product surface.

    Prefer explicit args; fall back to request ContextVars only when needed.
    """
    feat = (feature or "").strip().lower()
    mode = (agent_mode if agent_mode is not None else current_agent_mode.get() or "") or ""
    mode = str(mode).strip().lower()

    # Chapter / voice LLM features are product surfaces of their own.
    if voice_mode or feat == "voice":
        return "ai_voice_tutor"
    if feat == "chat":
        return "learn_mastro_text"

    if feat == "assistant" or mode in ("free", "ask", "practice", "explain"):
        if mode == "ask":
            return "ask_questions"
        if mode == "practice":
            return "practice_problems"
        if mode == "explain":
            return "explain_topic"
        # free / default assistant
        return "ask_learn_mastro"

    bound = current_product_surface.get()
    if bound in SURFACE_LABELS:
        return bound
    return "learn_mastro_text"


def touch_user_activity(
    *,
    user_id: int | None,
    feature: str,
    role: str,
    provider: str,
    model: str | None,
    api_key: str,
    api_key_index: int | None,
    key_tail: str,
    agent_mode: str | None = None,
    voice_mode: bool = False,
    surface: str | None = None,
) -> None:
    """Record that this user is actively using a surface with a specific LLM route."""
    if not user_id:
        return
    surf = surface or resolve_surface(feature=feature, agent_mode=agent_mode, voice_mode=voice_mode)
    now = time.time()
    with _lock:
        _user_activity[int(user_id)] = {
            "user_id": int(user_id),
            "surface": surf,
            "surface_label": SURFACE_LABELS.get(surf, surf),
            "feature": feature,
            "agent_mode": agent_mode or current_agent_mode.get(),
            "role": role,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "api_key_index": api_key_index,
            "key_tail": key_tail,
            "ts": now,
            "at": datetime.now(UTC).isoformat(),
        }
        _prune_locked(now)


def _prune_locked(now: float | None = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - ACTIVE_WINDOW_SEC
    stale = [uid for uid, row in _user_activity.items() if float(row.get("ts") or 0) < cutoff]
    for uid in stale:
        _user_activity.pop(uid, None)


def activity_snapshot(*, active_seconds: int = ACTIVE_WINDOW_SEC) -> dict[str, Any]:
    """Active users by product surface + recent per-user LLM routes."""
    now = time.time()
    cutoff = now - max(60, active_seconds)
    with _lock:
        _prune_locked(now)
        rows = [dict(v) for v in _user_activity.values() if float(v.get("ts") or 0) >= cutoff]

    # Enrich with username / full_name (fail-open)
    by_id: dict[int, dict[str, Any]] = {int(r["user_id"]): r for r in rows}
    if by_id:
        try:
            from sqlalchemy import select

            from app.core.database import SessionLocal
            from app.modules.users.models import User, UserSettings

            db = SessionLocal()
            try:
                users = {
                    int(u.id): u
                    for u in db.scalars(select(User).where(User.id.in_(list(by_id.keys())))).all()
                }
                settings = {
                    int(s.user_id): s
                    for s in db.scalars(
                        select(UserSettings).where(UserSettings.user_id.in_(list(by_id.keys())))
                    ).all()
                }
                for uid, row in by_id.items():
                    u = users.get(uid)
                    s = settings.get(uid)
                    row["username"] = (s.username if s and s.username else None) or (u.email if u else None)
                    row["full_name"] = u.full_name if u else None
                    row["email"] = u.email if u else None
            finally:
                db.close()
        except Exception as exc:
            logger.warning("activity username enrich failed: %s", exc)

    active_by_surface: dict[str, list[dict[str, Any]]] = {k: [] for k in SURFACE_LABELS}
    for row in by_id.values():
        surf = str(row.get("surface") or "")
        if surf in active_by_surface:
            active_by_surface[surf].append(row)

    counts = {
        key: {
            "surface": key,
            "label": SURFACE_LABELS[key],
            "active_users": len(active_by_surface[key]),
            "users": [
                {
                    "user_id": u.get("user_id"),
                    "username": u.get("username"),
                    "full_name": u.get("full_name"),
                    "role": u.get("role"),
                    "provider": u.get("provider"),
                    "model": u.get("model"),
                    "api_key": u.get("api_key"),
                    "api_key_index": u.get("api_key_index"),
                    "at": u.get("at"),
                }
                for u in sorted(active_by_surface[key], key=lambda x: float(x.get("ts") or 0), reverse=True)
            ],
        }
        for key in SURFACE_LABELS
    }

    recent_users = sorted(by_id.values(), key=lambda x: float(x.get("ts") or 0), reverse=True)
    return {
        "active_window_seconds": active_seconds,
        "active_total": len(by_id),
        "surfaces": list(counts.values()),
        "recent_users": [
            {
                "user_id": u.get("user_id"),
                "username": u.get("username"),
                "full_name": u.get("full_name"),
                "email": u.get("email"),
                "surface": u.get("surface"),
                "surface_label": u.get("surface_label"),
                "feature": u.get("feature"),
                "agent_mode": u.get("agent_mode"),
                "role": u.get("role"),
                "provider": u.get("provider"),
                "model": u.get("model"),
                "api_key": u.get("api_key"),
                "api_key_index": u.get("api_key_index"),
                "key_tail": u.get("key_tail"),
                "at": u.get("at"),
            }
            for u in recent_users[:100]
        ],
    }

"""Attribute provider-reported LLM token usage to the signed-in caller."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any
from urllib.parse import parse_qs

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

current_user_id: ContextVar[int | None] = ContextVar("llm_usage_user_id", default=None)


def _user_id_from_scope(scope: Scope) -> int | None:
    from app.core.security import decode_token

    raw = ""
    for key, value in scope.get("headers") or []:
        if key.lower() == b"authorization":
            auth = value.decode("latin-1")
            if auth.lower().startswith("bearer "):
                raw = auth[7:].strip()
            break
    if not raw:
        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        raw = (qs.get("access_token") or qs.get("token") or [""])[0]
    if not raw:
        return None
    try:
        payload = decode_token(raw)
        return int(payload["sub"]) if payload.get("type") == "access" else None
    except Exception:
        return None


class UsageUserMiddleware:
    """Sets `current_user_id` for HTTP + WebSocket requests (tasks spawned inside inherit it)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        reset = current_user_id.set(_user_id_from_scope(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(reset)


def usage_from(payload: dict[str, Any]) -> tuple[int, int] | None:
    """(prompt, completion) from an OpenAI-compatible body or stream chunk (Groq nests it under x_groq)."""
    usage = payload.get("usage") or (payload.get("x_groq") or {}).get("usage")
    if not isinstance(usage, dict):
        return None
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def record_usage(user_id: int | None, feature: str, model: str, prompt: int, completion: int) -> None:
    """Never raises — a telemetry failure must not break the answer the user is waiting for."""
    from app.core.database import SessionLocal
    from app.modules.users.models import LlmUsage

    db = SessionLocal()
    try:
        db.add(
            LlmUsage(
                user_id=user_id,
                feature=feature[:40],
                model=model[:120],
                prompt_tokens=prompt,
                completion_tokens=completion,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("failed to record LLM usage feature=%s", feature)
    finally:
        db.close()


def _self_check() -> None:
    assert usage_from({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}) == (7, 3)
    assert usage_from({"x_groq": {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}}) == (1, 2)
    assert usage_from({"choices": []}) is None


_self_check()

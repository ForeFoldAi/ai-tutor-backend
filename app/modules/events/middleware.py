from __future__ import annotations

import logging
import re

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.database import SessionLocal
from app.core.security import decode_token
from app.modules.events.hub import publish_domain_event
from app.modules.users.models import User

logger = logging.getLogger(__name__)

# Mutating API path → domain entity for live invalidation.
_PATH_ENTITY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/auth/admin/students(?:/|$)"), "students"),
    (re.compile(r"^/auth/tutor/students(?:/|$)"), "students"),
    (re.compile(r"^/auth/admin/teachers(?:/|$)"), "teachers"),
    (re.compile(r"^/auth/admin/classes(?:/|$)"), "classes"),
    (re.compile(r"^/auth/admin/credentials(?:/|$)"), "credentials"),
    (re.compile(r"^/auth/admin/dashboard(?:/|$)"), "dashboard"),
    (re.compile(r"^/auth/admin/users(?:/|$)"), "users"),
    (re.compile(r"^/auth/admin/schools(?:/|$)"), "schools"),
    (re.compile(r"^/auth/admin/create-school-admin(?:/|$)"), "users"),
    (re.compile(r"^/auth/tutor/live-sessions(?:/|$)"), "sessions"),
    (re.compile(r"^/auth/student/live-sessions(?:/|$)"), "sessions"),
    (re.compile(r"^/api/tutor/assignments(?:/|$)"), "assignments"),
    (re.compile(r"^/api/student/assignments(?:/|$)"), "assignments"),
    (re.compile(r"^/auth/me/notifications(?:/|$)"), "notifications"),
    (re.compile(r"^/auth/master/notifications(?:/|$)"), "notifications"),
]

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def _entity_for_path(path: str) -> str | None:
    for pattern, entity in _PATH_ENTITY:
        if pattern.search(path):
            return entity
    return None


def _actor_from_headers(headers: list[tuple[bytes, bytes]]) -> tuple[int | None, int | None]:
    auth = ""
    for key, value in headers:
        if key.lower() == b"authorization":
            auth = value.decode("latin-1")
            break
    if not auth.lower().startswith("bearer "):
        return None, None
    token = auth[7:].strip()
    if not token:
        return None, None
    try:
        payload = decode_token(token)
    except ValueError:
        return None, None
    if payload.get("type") != "access":
        return None, None
    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        return None, None

    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if user is None:
            return user_id, None
        return user.id, user.school_id
    finally:
        db.close()


class DomainEventMiddleware:
    """Pure ASGI middleware — HTTP only, so /ws/* is untouched."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        entity = _entity_for_path(path) if method in _MUTATING else None
        status_code_holder = {"code": 0}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code_holder["code"] = int(message.get("status", 0))
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if entity is None:
            return
        if status_code_holder["code"] >= 400 or status_code_holder["code"] == 0:
            return

        actor_id, school_id = _actor_from_headers(list(scope.get("headers") or []))
        query = scope.get("query_string", b"").decode("latin-1")
        for part in query.split("&"):
            if part.startswith("school_id=") and part[10:].isdigit():
                school_id = int(part[10:])
                break

        try:
            publish_domain_event(
                entity=entity,
                action=method.lower(),
                school_id=school_id,
                actor_id=actor_id,
            )
        except Exception:
            logger.exception("failed to publish domain event for %s", entity)

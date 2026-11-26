"""One active login per student, per browser.

A login gets a ``sid`` that rides in the access token and on every rotated refresh row. The
user row holds the one ``active_sid`` allowed right now; any other token is rejected, which is
what signs the old device out (access tokens are otherwise stateless for 15 minutes).

A new login is refused (409) while a valid login from a *different browser* exists, unless the
student confirms. A browser is identified by ``device_id`` (random, kept in localStorage), so the
same browser signing in again is not treated as another device.

Only students are limited. ``SINGLE_SESSION_ENABLED=false`` turns the whole thing off.
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import status
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.student_messages import ALREADY_SIGNED_IN_ELSEWHERE, SESSION_REPLACED
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User

settings = get_settings()

# WebSocket close code the frontend reads to sign out with the "opened on another device" message.
WS_CLOSE_SESSION_REPLACED = 4409


def applies_to(user: User) -> bool:
    return settings.single_session_enabled and user.role == Role.STUDENT


def new_sid() -> str:
    return uuid4().hex


def sid_is_current(user: User, sid: str | None) -> bool:
    """False only when this student has a current login and ``sid`` is not it.

    Users with no ``active_sid`` (never logged in since this shipped, or logged out) are not
    enforced, so tokens issued before the rollout keep working until the next login.
    """
    if not applies_to(user) or user.active_sid is None:
        return True
    return sid == user.active_sid


def assert_session_current(user: User, payload: dict) -> None:
    if not sid_is_current(user, payload.get("sid")):
        raise AuthException(SESSION_REPLACED, status.HTTP_401_UNAUTHORIZED)


def payload_session_replaced(payload: dict) -> bool:
    """For endpoints that decode the bearer themselves (voice): has this student signed in elsewhere?

    Blocking (own DB session) — call it through ``asyncio.to_thread`` from async code.
    """
    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        return False
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        return user is not None and not sid_is_current(user, payload.get("sid"))
    finally:
        db.close()


def _has_other_device_session(db: Session, user_id: int, device_id: str | None) -> bool:
    """A valid login from a browser other than ``device_id``.

    Logins created before device ids existed (NULL) count as another device.
    """
    conditions = [
        SessionToken.user_id == user_id,
        SessionToken.revoked.is_(False),
        SessionToken.expires_at > datetime.now(UTC),
    ]
    if device_id is not None:
        conditions.append(or_(SessionToken.device_id.is_(None), SessionToken.device_id != device_id))
    row = db.scalar(select(SessionToken.id).where(*conditions).limit(1))
    return row is not None


def start_session(db: Session, user: User, *, force: bool, device_id: str | None = None) -> str:
    """Begin a student's login. Raises 409 if they are signed in on another browser and ``force`` is false.

    On success every other login is revoked and the returned sid becomes the only valid one.
    """
    # Serialise concurrent logins for this user so two devices can't both win.
    db.scalar(select(User).where(User.id == user.id).with_for_update().execution_options(populate_existing=True))
    if not force and _has_other_device_session(db, user.id, device_id):
        raise AuthException(ALREADY_SIGNED_IN_ELSEWHERE, status.HTTP_409_CONFLICT)
    db.execute(
        update(SessionToken)
        .where(SessionToken.user_id == user.id, SessionToken.revoked.is_(False))
        .values(revoked=True)
    )
    sid = new_sid()
    user.active_sid = sid
    return sid


def end_session(user: User, sid: str | None, *, all_devices: bool = False) -> None:
    """Logout: free the slot, but only if the device logging out is the current one."""
    if not applies_to(user):
        return
    if all_devices or (sid is not None and sid == user.active_sid):
        user.active_sid = None

from __future__ import annotations

import re

from fastapi import status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.signup.validators import (
    find_user_by_username,
    normalize_phone,
    normalize_username,
)
from app.modules.users.models import User, UserSettings

_USERNAME_SAFE = re.compile(r"[^a-z0-9._-]+")


def resolve_target_school_id(
    actor: User,
    *,
    school_id: int | None = None,
) -> int:
    if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR):
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        return actor.school_id
    if actor.role == Role.MASTER_ADMIN:
        if school_id is None:
            raise AuthException("school_id is required.", status.HTTP_400_BAD_REQUEST)
        return school_id
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def assert_actor_manages_teacher(actor: User, teacher: User) -> None:
    if teacher.role != Role.TUTOR:
        raise AuthException("Not a teacher account.", status.HTTP_400_BAD_REQUEST)
    if actor.role == Role.MASTER_ADMIN:
        return
    if actor.role == Role.SCHOOL_ADMIN:
        if teacher.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def generate_unique_username(db: Session, full_name: str, email: str) -> str:
    local = email.split("@", 1)[0].lower()
    base = _USERNAME_SAFE.sub("", local.replace(" ", "."))
    if not base:
        parts = [p for p in full_name.lower().split() if p]
        base = _USERNAME_SAFE.sub("", ".".join(parts[:2]) if parts else "teacher")
    if not base:
        base = "teacher"

    candidate = base[:48]
    suffix = 1
    while find_user_by_username(db, candidate):
        tail = str(suffix)
        candidate = f"{base[: max(1, 48 - len(tail))]}{tail}"
        suffix += 1
    return candidate


def username_taken(db: Session, username: str, exclude_user_id: int | None = None) -> bool:
    ident = normalize_username(username)
    if not ident:
        return True
    stmt = select(UserSettings).where(func.lower(UserSettings.username) == ident)
    if exclude_user_id:
        stmt = stmt.where(UserSettings.user_id != exclude_user_id)
    return db.scalar(stmt) is not None

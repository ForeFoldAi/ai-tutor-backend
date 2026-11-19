from __future__ import annotations

from datetime import UTC, datetime

from fastapi import status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.events.hub import publish_domain_event
from app.modules.notifications.models import Notification
from app.modules.notifications.schemas import (
    MasterNotifyRequest,
    MasterNotifyResponse,
    MarkReadResponse,
    NotificationListResponse,
    NotificationOut,
    UnreadCountResponse,
)
from app.modules.users.models import User

LIST_LIMIT = 50


def notify_users(
    db: Session,
    *,
    recipient_ids: list[int] | set[int],
    type: str,
    title: str,
    body: str = "",
    actor_id: int | None = None,
    school_id: int | None = None,
    link: str | None = None,
) -> int:
    """Bulk-insert inbox rows. Dedupes recipient ids. Does not commit."""
    ids = list(dict.fromkeys(int(i) for i in recipient_ids if i is not None))
    if not ids:
        return 0
    now = datetime.now(UTC)
    rows = [
        Notification(
            recipient_user_id=rid,
            actor_user_id=actor_id,
            type=type,
            title=title[:200],
            body=body or "",
            link=link,
            school_id=school_id,
            created_at=now,
        )
        for rid in ids
    ]
    db.add_all(rows)
    db.flush()
    publish_domain_event(
        entity="notifications",
        action="create",
        school_id=school_id,
        actor_id=actor_id,
        payload={"recipient_ids": ids},
    )
    return len(rows)


def list_my_notifications(db: Session, user: User, *, limit: int = LIST_LIMIT) -> NotificationListResponse:
    lim = max(1, min(limit, 100))
    rows = list(
        db.scalars(
            select(Notification)
            .where(Notification.recipient_user_id == user.id)
            .order_by(Notification.created_at.desc())
            .limit(lim)
        )
    )
    unread = unread_count(db, user).count
    return NotificationListResponse(
        items=[NotificationOut.model_validate(r) for r in rows],
        unread_count=unread,
    )


def unread_count(db: Session, user: User) -> UnreadCountResponse:
    count = db.scalar(
        select(func.count())
        .select_from(Notification)
        .where(
            Notification.recipient_user_id == user.id,
            Notification.read_at.is_(None),
        )
    )
    return UnreadCountResponse(count=int(count or 0))


def mark_read(db: Session, user: User, ids: list[int] | None) -> MarkReadResponse:
    now = datetime.now(UTC)
    stmt = (
        update(Notification)
        .where(
            Notification.recipient_user_id == user.id,
            Notification.read_at.is_(None),
        )
        .values(read_at=now)
    )
    if ids:
        stmt = stmt.where(Notification.id.in_(list(dict.fromkeys(ids))))
    result = db.execute(stmt)
    db.flush()
    updated = int(result.rowcount or 0)
    if updated:
        publish_domain_event(
            entity="notifications",
            action="mark_read",
            school_id=user.school_id,
            actor_id=user.id,
            payload={"recipient_ids": [user.id]},
        )
    return MarkReadResponse(updated=updated)


def master_notify(db: Session, actor: User, payload: MasterNotifyRequest) -> MasterNotifyResponse:
    if actor.role != Role.MASTER_ADMIN:
        raise AuthException("Only master admin can send these notifications.", status.HTTP_403_FORBIDDEN)

    title = payload.title.strip()
    body = payload.body.strip()
    if not title or not body:
        raise AuthException("Title and body are required.", status.HTTP_400_BAD_REQUEST)

    want = list(dict.fromkeys(payload.recipient_ids))
    admins = list(
        db.scalars(
            select(User).where(
                User.id.in_(want),
                User.role == Role.SCHOOL_ADMIN,
                User.is_active.is_(True),
            )
        )
    )
    if len(admins) != len(want):
        raise AuthException(
            "All recipients must be active school administrators.",
            status.HTTP_400_BAD_REQUEST,
        )

    # One school_id when all share it; else None (still fans out via user rooms).
    school_ids = {a.school_id for a in admins if a.school_id is not None}
    school_id = next(iter(school_ids)) if len(school_ids) == 1 else None

    created = notify_users(
        db,
        recipient_ids=[a.id for a in admins],
        type="master_message",
        title=title,
        body=body,
        actor_id=actor.id,
        school_id=school_id,
    )
    return MasterNotifyResponse(created=created)

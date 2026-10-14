from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user, require_roles
from app.modules.notifications import service as notify_service
from app.modules.notifications.schemas import (
    MasterNotifyRequest,
    MasterNotifyResponse,
    MarkReadRequest,
    MarkReadResponse,
    NotificationListResponse,
    UnreadCountResponse,
)
from app.modules.users.models import User

router = APIRouter(tags=["notifications"])

CurrentUser = Annotated[User, Depends(get_current_user)]
Db = Annotated[Session, Depends(get_db)]


@router.get("/auth/me/notifications", response_model=NotificationListResponse)
def list_notifications(
    db: Db,
    current_user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    return notify_service.list_my_notifications(db, current_user, limit=limit)


@router.get("/auth/me/notifications/unread-count", response_model=UnreadCountResponse)
def get_unread_count(db: Db, current_user: CurrentUser):
    return notify_service.unread_count(db, current_user)


@router.post("/auth/me/notifications/mark-read", response_model=MarkReadResponse)
def post_mark_read(payload: MarkReadRequest, db: Db, current_user: CurrentUser):
    result = notify_service.mark_read(db, current_user, payload.ids)
    db.commit()
    return result


@router.post(
    "/auth/master/notifications",
    response_model=MasterNotifyResponse,
    status_code=201,
)
def post_master_notify(
    payload: MasterNotifyRequest,
    db: Db,
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    result = notify_service.master_notify(db, current_user, payload)
    db.commit()
    return result

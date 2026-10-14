from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class NotificationOut(BaseModel):
    id: int
    type: str
    title: str
    body: str
    link: str | None = None
    read_at: datetime | None = None
    created_at: datetime
    actor_user_id: int | None = None

    model_config = {"from_attributes": True}


class NotificationListResponse(BaseModel):
    items: list[NotificationOut]
    unread_count: int


class UnreadCountResponse(BaseModel):
    count: int


class MarkReadRequest(BaseModel):
    ids: list[int] | None = None


class MarkReadResponse(BaseModel):
    updated: int


class MasterNotifyRequest(BaseModel):
    recipient_ids: list[int] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=2000)


class MasterNotifyResponse(BaseModel):
    created: int

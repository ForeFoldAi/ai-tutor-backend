from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LiveSessionCreateRequest(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    subject: str = Field(min_length=1, max_length=120)
    chapter_id: str | None = Field(default=None, max_length=64)
    chapter: str | None = Field(default=None, max_length=150)
    grade: str = Field(min_length=1, max_length=20)
    section: str = Field(min_length=1, max_length=20)
    curriculum: str | None = Field(default=None, max_length=100)
    starts_at: datetime
    duration_minutes: int = Field(default=60, ge=15, le=240)
    meeting_link: str | None = Field(default=None, max_length=512)
    notes: str | None = None

    @field_validator("title", "subject", "grade", "section", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("meeting_link", mode="before")
    @classmethod
    def _clean_link(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("Meeting link must start with http:// or https://")
        return s


class LiveSessionUpdateRequest(LiveSessionCreateRequest):
    """Same fields as create — full replace of editable session details."""


class LiveSessionOut(BaseModel):
    id: int
    title: str
    subject: str
    chapter_id: str | None = None
    chapter: str | None = None
    grade: str
    section: str
    curriculum: str | None = None
    starts_at: datetime
    duration_minutes: int
    meeting_link: str | None = None
    notes: str | None = None
    tutor_id: int
    tutor_name: str
    attendees: int = 0
    status: Literal["live", "upcoming", "completed"]
    joined: bool = False

    model_config = {"from_attributes": True}


class LiveSessionJoinResponse(BaseModel):
    session_id: int
    meeting_link: str
    joined_at: datetime

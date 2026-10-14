from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    default_limit: int


class CredentialRecord(BaseModel):
    id: int
    name: str
    role: str
    user_id: str
    grade: str | None = None
    section: str | None = None
    curriculum: str | None = None
    has_credentials: bool = False
    credential_shared: str | None
    first_login_status: str
    last_login: str | None
    delivery_status: str


class CredentialListResponse(BaseModel):
    items: list[CredentialRecord]
    meta: PaginationMeta


class CredentialMetrics(BaseModel):
    generated: int
    teachers_pending_login: int
    students_pending_login: int
    not_shared: int
    generated_trend: str
    teachers_pending_trend: str
    students_pending_trend: str
    not_shared_trend: str
    not_shared_trend_up: bool


class CredentialCandidate(BaseModel):
    id: int
    name: str
    role: str
    user_id: str
    grade: str | None = None
    section: str | None = None
    detail: str | None = None
    has_credentials: bool


class CredentialCandidatesResponse(BaseModel):
    items: list[CredentialCandidate]
    grades: list[str]
    sections: list[str]


class CredentialActionRequest(BaseModel):
    role: str = Field(description="Teacher | Student")
    user_ids: list[int] = Field(min_length=1, max_length=500)


class CredentialGeneratedItem(BaseModel):
    id: int
    name: str
    role: str
    user_id: str
    password: str


class CredentialGenerateResponse(BaseModel):
    message: str
    items: list[CredentialGeneratedItem]


class CredentialSendResponse(BaseModel):
    message: str
    updated: int

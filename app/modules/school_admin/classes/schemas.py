from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    default_limit: int


class ClassOptionsResponse(BaseModel):
    curricula: list[str]
    default_limit: int
    page_size_options: list[int]


class SubjectCreateItem(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    code: str = Field(min_length=1, max_length=32)

    @field_validator("name", "code", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("This field is required.")
        return s

    @field_validator("code", mode="after")
    @classmethod
    def _upper_code(cls, v: str) -> str:
        return v.upper()


class SubjectUpdateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    code: str = Field(min_length=1, max_length=32)

    @field_validator("name", "code", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("This field is required.")
        return s

    @field_validator("code", mode="after")
    @classmethod
    def _upper_code(cls, v: str) -> str:
        return v.upper()

class SubjectBulkCreateRequest(BaseModel):
    subjects: list[SubjectCreateItem] = Field(min_length=1, max_length=200)


class SubjectResponse(BaseModel):
    id: int
    name: str
    code: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class SubjectListResponse(BaseModel):
    items: list[SubjectResponse]
    meta: PaginationMeta


class ClassCreateItem(BaseModel):
    grade: str = Field(min_length=1, max_length=20)
    section: str = Field(min_length=1, max_length=20)
    curriculum: str = Field(min_length=1, max_length=100)

    @field_validator("grade", "section", "curriculum", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("This field is required.")
        return s

    @field_validator("section", mode="after")
    @classmethod
    def _upper_section(cls, v: str) -> str:
        return v.upper()


class ClassUpdateRequest(BaseModel):
    grade: str = Field(min_length=1, max_length=20)
    section: str = Field(min_length=1, max_length=20)
    curriculum: str = Field(min_length=1, max_length=100)

    @field_validator("grade", "section", "curriculum", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("This field is required.")
        return s

    @field_validator("section", mode="after")
    @classmethod
    def _upper_section(cls, v: str) -> str:
        return v.upper()


class ClassBulkCreateRequest(BaseModel):
    classes: list[ClassCreateItem] = Field(min_length=1, max_length=200)


class ClassResponse(BaseModel):
    id: int
    grade: str
    section: str
    curriculum: str
    students: int
    teachers: int
    created_at: datetime
    updated_at: datetime


class ClassListResponse(BaseModel):
    items: list[ClassResponse]
    meta: PaginationMeta


class BulkRowError(BaseModel):
    row: int
    field: str | None = None
    message: str


class SubjectBulkCreateResponse(BaseModel):
    created: list[SubjectResponse]
    errors: list[BulkRowError]


class ClassBulkCreateResponse(BaseModel):
    created: list[ClassResponse]
    errors: list[BulkRowError]


class ClassSubjectMappingSaveRequest(BaseModel):
    subject_ids: list[int] = Field(default_factory=list)


class ClassSubjectMappingsResponse(BaseModel):
    mappings: dict[str, dict[str, bool]]

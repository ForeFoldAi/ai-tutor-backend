from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    default_limit: int


class StudentCreateItem(BaseModel):
    """Add/import row: roll_number → username (user id)."""

    roll_number: str = Field(min_length=1, max_length=64)
    student_name: str = Field(min_length=2, max_length=255)
    parent_phone: str = Field(min_length=6, max_length=50)
    parent_email: EmailStr

    @field_validator("roll_number", "student_name", "parent_phone", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("This field is required.")
        return s


class StudentBulkCreateRequest(BaseModel):
    students: list[StudentCreateItem] = Field(min_length=1, max_length=200)


class StudentUpdateRequest(BaseModel):
    student_name: str | None = Field(default=None, min_length=2, max_length=255)
    parent_phone: str | None = Field(default=None, min_length=6, max_length=50)
    parent_email: EmailStr | None = None
    is_active: bool | None = None
    grade: str | None = Field(default=None, max_length=32)
    section: str | None = Field(default=None, max_length=16)
    curriculum: str | None = Field(default=None, max_length=100)
    class_id: int | None = None

    @field_validator("student_name", "parent_phone", "grade", "section", "curriculum", mode="before")
    @classmethod
    def _strip_optional(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class StudentResponse(BaseModel):
    id: int
    user_id: str
    full_name: str
    email: EmailStr
    phone: str | None
    school_id: int | None
    grade: str | None
    section: str | None
    curriculum: str | None
    learning_type: str
    learning_teacher: str | None
    learning_teacher_ids: list[int] = Field(default_factory=list)
    """Class subjects covered by a matched teacher."""
    teacher_guided_subjects: list[str] = Field(default_factory=list)
    """Class subjects with no matched teacher — Self learned."""
    self_learned_subjects: list[str] = Field(default_factory=list)
    password_status: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class StudentListResponse(BaseModel):
    items: list[StudentResponse]
    meta: PaginationMeta


class StudentOptionsResponse(BaseModel):
    grades: list[str]
    sections: list[str]
    curricula: list[str]
    learning_types: list[str]
    default_limit: int
    page_size_options: list[int]


class BulkRowError(BaseModel):
    row: int
    email: str | None = None
    roll_number: str | None = None
    message: str


class StudentBulkCreateResponse(BaseModel):
    created: list[StudentResponse]
    errors: list[BulkRowError]


class StudentAssignTeacherRequest(BaseModel):
    teacher_ids: list[int] = Field(min_length=1, max_length=200)
    student_ids: list[int] = Field(min_length=1, max_length=500)


class StudentMoveClassRequest(BaseModel):
    student_ids: list[int] = Field(min_length=1, max_length=500)
    class_id: int


class StudentActionResponse(BaseModel):
    message: str
    updated: int

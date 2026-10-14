from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    default_limit: int


class TeacherCreateItem(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    phone: str = Field(min_length=6, max_length=50)
    email: EmailStr

    @field_validator("full_name", "phone", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("This field is required.")
        return s


class TeacherBulkCreateRequest(BaseModel):
    teachers: list[TeacherCreateItem] = Field(min_length=1, max_length=200)


class TeacherUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=255)
    phone: str | None = Field(default=None, min_length=6, max_length=50)
    email: EmailStr | None = None
    is_active: bool | None = None
    teaching_board: str | None = Field(default=None, max_length=100)

    @field_validator("full_name", "phone", "teaching_board", mode="before")
    @classmethod
    def _strip_optional(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class TeacherGradeSubjects(BaseModel):
    """One grade/class row with subjects taught in that class (for nested table view)."""

    grade: str
    subjects: str


class TeacherResponse(BaseModel):
    id: int
    user_id: str
    full_name: str
    email: EmailStr
    phone: str | None
    school_id: int | None
    subject: str | None
    grades: str | None
    assignments: list[TeacherGradeSubjects] = Field(default_factory=list)
    assigned_students: int
    is_active: bool
    last_login: datetime | None
    created_at: datetime
    updated_at: datetime


class TeacherSubjectBrief(BaseModel):
    id: int
    name: str
    code: str


class TeacherClassBrief(BaseModel):
    id: int
    grade: str
    section: str
    curriculum: str


class TeacherStudentBrief(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    grade: str | None = None
    section: str | None = None


class TeacherDetailResponse(TeacherResponse):
    subjects: list[TeacherSubjectBrief] = Field(default_factory=list)
    classes: list[TeacherClassBrief] = Field(default_factory=list)
    students: list[TeacherStudentBrief] = Field(default_factory=list)


class TeacherUnassignRequest(BaseModel):
    subject_ids: list[int] = Field(default_factory=list)
    class_ids: list[int] = Field(default_factory=list)
    student_ids: list[int] = Field(default_factory=list)


class TeacherListResponse(BaseModel):
    items: list[TeacherResponse]
    meta: PaginationMeta


class TeacherOptionsResponse(BaseModel):
    subjects: list[str]
    default_limit: int
    page_size_options: list[int]


class BulkRowError(BaseModel):
    row: int
    email: str | None = None
    message: str


class TeacherBulkCreateResponse(BaseModel):
    created: list[TeacherResponse]
    errors: list[BulkRowError]


class TeacherAssignClassesRequest(BaseModel):
    teacher_ids: list[int] = Field(min_length=1, max_length=200)
    class_ids: list[int] = Field(min_length=1, max_length=200)


class TeacherAssignSubjectsRequest(BaseModel):
    teacher_ids: list[int] = Field(min_length=1, max_length=200)
    subject_ids: list[int] = Field(min_length=1, max_length=200)


class TeacherAssignResponse(BaseModel):
    updated: int
    message: str

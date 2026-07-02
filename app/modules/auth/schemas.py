import uuid
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.modules.auth.constants import Role


class StudentSignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    school_name: str = Field(min_length=2, max_length=255)
    grade: str = Field(min_length=1, max_length=50)
    board: str = Field(min_length=1, max_length=100)


class OrganizationSignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    organization_name: str = Field(min_length=2, max_length=255)
    phone: str | None = Field(default=None, max_length=50)
    address: str | None = Field(default=None, max_length=500)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None
    all_devices: bool = False


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


class AdminCreateUserRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    organization_id: uuid.UUID | None = None
    school_id: uuid.UUID | None = None


class TeachingClassAssignment(BaseModel):
    """A class level (grade 1–10, etc.) and the sections the tutor covers."""

    grade: str = Field(min_length=1, max_length=50, description="Class / grade label, e.g. 1–10")
    sections: list[str] = Field(min_length=1, max_length=24)

    @field_validator("grade", mode="before")
    @classmethod
    def _strip_grade(cls, v: object) -> str:
        if v is None:
            raise ValueError("Please select your class.")
        s = str(v).strip()
        if not s:
            raise ValueError("Please select your class.")
        return s

    @field_validator("sections", mode="before")
    @classmethod
    def _normalize_sections(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("sections must be a list")
        out: list[str] = []
        for x in v:
            s = str(x).strip().upper()
            if s:
                out.append(s[:20])
        if not out:
            raise ValueError("Please enter your section (for example A or B).")
        return out[:24]


class CreateTutorRequest(AdminCreateUserRequest):
    """Onboard a tutor with school assignment and class/section scope."""

    teaching_board: str | None = Field(default=None, max_length=100)
    teaching_classes: list[TeachingClassAssignment] = Field(min_length=1, max_length=40)

    @field_validator("teaching_board", mode="before")
    @classmethod
    def _blank_board(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class UpdateTutorRequest(BaseModel):
    """Update an existing tutor (profile, school, classes/sections, optional password reset)."""

    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    school_id: uuid.UUID
    teaching_board: str | None = Field(default=None, max_length=100)
    teaching_classes: list[TeachingClassAssignment] = Field(min_length=1, max_length=40)
    new_password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("teaching_board", mode="before")
    @classmethod
    def _blank_board(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("new_password", mode="before")
    @classmethod
    def _empty_password(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class CreateStudentRequest(AdminCreateUserRequest):
    """Onboard a student: one grade (class) and exactly one section."""

    teaching_board: str | None = Field(default=None, max_length=100)
    teaching_classes: list[TeachingClassAssignment] = Field(min_length=1, max_length=1)

    @field_validator("teaching_board", mode="before")
    @classmethod
    def _blank_board(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @model_validator(mode="after")
    def _single_section_only(self) -> Self:
        if len(self.teaching_classes[0].sections) != 1:
            raise ValueError("Please enter your section (for example A or B).")
        return self


class UpdateStudentRequest(BaseModel):
    """Update a student: one grade and exactly one section."""

    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    school_id: uuid.UUID
    teaching_board: str | None = Field(default=None, max_length=100)
    teaching_classes: list[TeachingClassAssignment] = Field(min_length=1, max_length=1)
    new_password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("teaching_board", mode="before")
    @classmethod
    def _blank_board(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("new_password", mode="before")
    @classmethod
    def _empty_password(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @model_validator(mode="after")
    def _single_section_only(self) -> Self:
        if len(self.teaching_classes[0].sections) != 1:
            raise ValueError("Please enter your section (for example A or B).")
        return self


class UserStatusPatchRequest(BaseModel):
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class MessageResponse(BaseModel):
    message: str


class UserResponse(BaseModel):
    id: uuid.UUID
    full_name: str
    email: EmailStr
    role: Role
    is_active: bool
    is_verified: bool
    organization_id: uuid.UUID | None
    school_id: uuid.UUID | None
    teaching_board: str | None = None
    teaching_classes: list[TeachingClassAssignment] | None = None
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _coerce_teaching_classes(cls, data: Any) -> Any:
        """Coerce legacy JSON list[str] to [{grade, sections}]; normalize dict inputs."""
        if data is None:
            return data
        if isinstance(data, dict):
            raw = data.get("teaching_classes")
            if isinstance(raw, list) and raw and isinstance(raw[0], str):
                return {
                    **data,
                    "teaching_classes": [{"grade": str(g), "sections": ["A"]} for g in raw if str(g).strip()],
                }
            return data
        if not hasattr(data, "teaching_classes"):
            return data
        inst = data
        raw = getattr(inst, "teaching_classes", None)
        if raw is None:
            return inst
        if isinstance(raw, list) and raw and isinstance(raw[0], str):
            coerced = [{"grade": str(g), "sections": ["A"]} for g in raw if str(g).strip()]
            return {
                "id": inst.id,
                "full_name": inst.full_name,
                "email": inst.email,
                "role": inst.role,
                "is_active": inst.is_active,
                "is_verified": inst.is_verified,
                "organization_id": inst.organization_id,
                "school_id": inst.school_id,
                "teaching_board": inst.teaching_board,
                "teaching_classes": coerced,
                "created_by": inst.created_by,
                "created_at": inst.created_at,
                "updated_at": inst.updated_at,
            }
        return inst


class SchoolAdminBrief(BaseModel):
    id: uuid.UUID
    full_name: str
    email: EmailStr
    is_active: bool

    model_config = {"from_attributes": True}


class SchoolSummaryResponse(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    organization_name: str | None = None
    name: str
    branch: str | None
    board: str | None
    created_at: datetime
    school_admins: list[SchoolAdminBrief]
    tutor_count: int
    student_count: int


class SchoolUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    branch: str | None = Field(default=None, max_length=255)
    board: str | None = Field(default=None, max_length=100)

    @field_validator("branch", "board", mode="before")
    @classmethod
    def _blank_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class OrganizationDetailResponse(BaseModel):
    id: uuid.UUID
    name: str
    phone: str | None
    address: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class OrganizationUpdateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    phone: str | None = Field(default=None, max_length=50)
    address: str | None = Field(default=None, max_length=500)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: object) -> str:
        s = str(v).strip()
        if len(s) < 2:
            raise ValueError("name must be at least 2 characters.")
        return s

    @field_validator("phone", "address", mode="before")
    @classmethod
    def _blank_optional_org(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class MeProfileUpdateRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    current_password: str | None = Field(default=None, max_length=128)
    new_password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("full_name", mode="before")
    @classmethod
    def _strip_full_name(cls, v: object) -> str:
        s = str(v).strip()
        if len(s) < 2:
            raise ValueError("Please enter your full name (at least 2 letters).")
        return s

    @field_validator("current_password", "new_password", mode="before")
    @classmethod
    def _empty_password_fields(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @model_validator(mode="after")
    def _new_password_requires_current(self) -> Self:
        if self.new_password and not self.current_password:
            raise ValueError("Enter your current password before choosing a new one.")
        return self


class UserSettingsResponse(BaseModel):
    user_id: uuid.UUID
    username: str | None
    language: str
    theme: str
    notify_email: bool
    notify_push: bool
    notify_assignments: bool
    notify_sessions: bool
    notify_messages: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserSettingsUpdateRequest(BaseModel):
    username: str | None = Field(default=None, min_length=2, max_length=64)
    language: str | None = Field(default=None, min_length=2, max_length=32)
    theme: str | None = Field(default=None, pattern=r"^(light|dark)$")
    notify_email: bool | None = None
    notify_push: bool | None = None
    notify_assignments: bool | None = None
    notify_sessions: bool | None = None
    notify_messages: bool | None = None

    @field_validator("username", "language", mode="before")
    @classmethod
    def _strip_optional_str(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("username")
    @classmethod
    def _username_chars(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username can only use letters, numbers, hyphens (-), and underscores (_)."
            )
        return v

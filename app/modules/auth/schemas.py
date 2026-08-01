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



class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None
    all_devices: bool = False


class ForgotPasswordLookupRequest(BaseModel):
    email: EmailStr


class ForgotPasswordAccount(BaseModel):
    id: int
    full_name: str
    username: str


class ForgotPasswordLookupResponse(BaseModel):
    accounts: list[ForgotPasswordAccount]


class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    user_id: int


class VerifyResetOtpRequest(BaseModel):
    email: EmailStr
    user_id: int
    otp: str = Field(min_length=4, max_length=12)


class VerifyResetOtpResponse(BaseModel):
    message: str
    reset_token: str


class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str = Field(min_length=8, max_length=128)


class AdminCreateUserRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    # Public school id (PK).
    school_id: int | None = None


class TeachingClassAssignment(BaseModel):
    """A class level (grade 1–10, etc.) and the sections the tutor covers."""

    grade: str = Field(min_length=1, max_length=50, description="Class / grade label, e.g. 1–10")
    sections: list[str] = Field(min_length=1, max_length=24)
    curriculum: str | None = Field(default=None, max_length=100)
    school_class_id: int | None = None

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

    @field_validator("curriculum", mode="before")
    @classmethod
    def _strip_curriculum(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s[:100] if s else None


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
    school_id: int
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
    school_id: int
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
    id: int
    full_name: str
    email: EmailStr
    role: Role
    is_active: bool
    is_verified: bool
    school_id: int | None
    phone: str | None = None
    designation: str | None = None
    teaching_board: str | None = None
    teaching_subjects: list[str] | None = None
    teaching_classes: list[TeachingClassAssignment] | None = None
    student_grade: str | None = None
    curricula: list[str] | None = None
    parent_email: str | None = None
    favorite_subjects: list[str] | None = None
    learning_goals: list[str] | None = None
    preferred_learning_method: str | None = None
    created_by: int | None
    created_at: datetime
    updated_at: datetime


class SchoolAdminBrief(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    is_active: bool


class SchoolSummaryResponse(BaseModel):
    id: int
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
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=50)
    website: str | None = Field(default=None, max_length=500)
    address: str | None = Field(default=None, max_length=500)

    @field_validator("branch", "board", "phone", "website", "address", mode="before")
    @classmethod
    def _blank_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    @field_validator("email", mode="before")
    @classmethod
    def _blank_email(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        return s if s else None


class SchoolDetailResponse(BaseModel):
    id: int
    name: str
    branch: str | None
    board: str | None
    email: str | None
    phone: str | None
    website: str | None
    address: str | None
    grades_offered: str | None = None
    student_strength: str | None = None
    curricula: list[str] = Field(default_factory=list)
    is_active: bool
    created_at: datetime


class SchoolProfileUpdateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    branch: str | None = Field(default=None, max_length=255)
    board: str | None = Field(default=None, max_length=100)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=50)
    address: str | None = Field(default=None, max_length=500)
    website: str | None = Field(default=None, max_length=500)
    grades_offered: str | None = Field(default=None, max_length=100)
    student_strength: str | None = Field(default=None, max_length=100)
    curricula: list[str] | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: object) -> str:
        s = str(v).strip()
        if len(s) < 2:
            raise ValueError("name must be at least 2 characters.")
        return s

    @field_validator("branch", "board", "phone", "address", "website", "grades_offered", "student_strength", mode="before")
    @classmethod
    def _blank_optional_profile(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("email", mode="before")
    @classmethod
    def _blank_email(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        return s if s else None


class MeProfileUpdateRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    phone: str | None = Field(default=None, max_length=50)
    designation: str | None = Field(default=None, max_length=100)
    grade: str | None = Field(default=None, max_length=50)
    curricula: list[str] | None = None
    parent_email: EmailStr | None = None
    favorite_subjects: list[str] | None = None
    learning_goals: list[str] | None = None
    preferred_learning_method: str | None = Field(default=None, max_length=32)
    current_password: str | None = Field(default=None, max_length=128)
    new_password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("full_name", mode="before")
    @classmethod
    def _strip_full_name(cls, v: object) -> str:
        s = str(v).strip()
        if len(s) < 2:
            raise ValueError("Please enter your full name (at least 2 letters).")
        return s

    @field_validator("phone", "designation", "current_password", "new_password", mode="before")
    @classmethod
    def _empty_optional_fields(cls, v: str | None) -> str | None:
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
    user_id: int
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

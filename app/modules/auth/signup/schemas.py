"""Pydantic request/response models for public signup."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.modules.auth.signup.enums import (
    ClassSizeEnum,
    LearningMethodEnum,
    SchoolGradeRangeEnum,
    StudentStrengthEnum,
    TeachingExperienceEnum,
    TeachingModeEnum,
    TeachingSubjectEnum,
    TutorGradeRangeEnum,
    parse_enum,
    parse_enum_list,
)
from app.modules.auth.signup.validators import (
    normalize_grade,
    normalize_phone,
    normalize_username,
    parse_curricula,
    parse_learning_goals,
    parse_learning_method,
    parse_student_grade,
    parse_student_subjects,
)


class SignupResponse(BaseModel):
    """Returned after successful public signup."""

    id: int = Field(description="Sequential account number (1, 2, 3, …)")
    user_id: str = Field(description="Chosen login user ID / username")
    role: str
    message: str


class SignupOptionsResponse(BaseModel):
    """Dropdown / multi-select options for signup forms."""

    curricula: list[str]
    student_grades: list[str]
    student_subjects: list[str]
    learning_goals: list[str]
    learning_methods: list[str]
    tutor_subjects: list[str]
    teaching_experience: list[str]
    tutor_grade_ranges: list[str]
    class_sizes: list[str]
    teaching_modes: list[str]
    school_grade_ranges: list[str]
    student_strength: list[str]


class StudentSignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    grade: str = Field(min_length=1, max_length=50)
    user_id: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    curricula: list[str] = Field(min_length=1, max_length=6)
    email: EmailStr | None = None
    parent_phone: str | None = Field(default=None, max_length=50)
    parent_email: EmailStr | None = None
    favorite_subjects: list[str] = Field(default_factory=list, max_length=12)
    learning_goals: list[str] = Field(default_factory=list, max_length=12)
    preferred_learning_method: str | None = Field(default=None, max_length=32)

    @field_validator("full_name", mode="before")
    @classmethod
    def _strip_name(cls, v: object) -> str:
        return str(v).strip()

    @field_validator("grade", mode="before")
    @classmethod
    def _norm_grade(cls, v: object) -> str:
        grade = parse_student_grade(str(v))
        return grade.value

    @field_validator("user_id", mode="before")
    @classmethod
    def _norm_user_id(cls, v: object) -> str:
        u = normalize_username(str(v))
        if not u:
            raise ValueError("User ID is required.")
        if not u.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                "User ID can only use letters, numbers, hyphens, underscores, and dots."
            )
        return u

    @field_validator("curricula", mode="before")
    @classmethod
    def _norm_curricula(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError("curricula must be a list")
        return [c.value for c in parse_curricula([str(x) for x in v])]

    @field_validator("parent_phone", mode="before")
    @classmethod
    def _norm_parent_phone(cls, v: object) -> str | None:
        if v is None:
            return None
        return normalize_phone(str(v))

    @field_validator("favorite_subjects", mode="before")
    @classmethod
    def _norm_subjects(cls, v: object) -> list[str]:
        if not v:
            return []
        if not isinstance(v, list):
            raise ValueError("favorite_subjects must be a list")
        return [s.value for s in parse_student_subjects([str(x) for x in v])]

    @field_validator("learning_goals", mode="before")
    @classmethod
    def _norm_goals(cls, v: object) -> list[str]:
        if not v:
            return []
        if not isinstance(v, list):
            raise ValueError("learning_goals must be a list")
        return [g.value for g in parse_learning_goals([str(x) for x in v])]

    @field_validator("preferred_learning_method", mode="before")
    @classmethod
    def _norm_method(cls, v: object) -> str | None:
        method = parse_learning_method(None if v is None else str(v))
        return method.value if method else None

    @model_validator(mode="after")
    def _resolve_email(self) -> Self:
        if self.email:
            return self
        if self.parent_email:
            object.__setattr__(self, "email", self.parent_email)
            return self
        object.__setattr__(self, "email", f"{self.user_id}@signup.aitutor.app")
        return self


class TutorSignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    user_id: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    email: EmailStr
    mobile: str = Field(min_length=8, max_length=50)
    teaching_experience: str = Field(min_length=1, max_length=50)
    grades_teach: str = Field(min_length=1, max_length=80)
    class_size: str = Field(min_length=1, max_length=50)
    teaching_subjects: list[str] = Field(min_length=1, max_length=12)
    teaching_modes: list[str] = Field(min_length=1, max_length=3)
    bio: str | None = Field(default=None, max_length=200)

    @field_validator("full_name", mode="before")
    @classmethod
    def _strip_name(cls, v: object) -> str:
        return str(v).strip()

    @field_validator("user_id", mode="before")
    @classmethod
    def _norm_user_id(cls, v: object) -> str:
        u = normalize_username(str(v))
        if not u:
            raise ValueError("User ID is required.")
        if not u.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                "User ID can only use letters, numbers, hyphens, underscores, and dots."
            )
        return u

    @field_validator("mobile", mode="before")
    @classmethod
    def _norm_mobile(cls, v: object) -> str:
        p = normalize_phone(str(v))
        if not p or len(p) < 8:
            raise ValueError("Enter a valid mobile number.")
        return p

    @field_validator("teaching_experience", mode="before")
    @classmethod
    def _norm_experience(cls, v: object) -> str:
        return parse_enum(TeachingExperienceEnum, str(v), "teaching experience").value

    @field_validator("grades_teach", mode="before")
    @classmethod
    def _norm_grades(cls, v: object) -> str:
        return parse_enum(TutorGradeRangeEnum, str(v), "grade range").value

    @field_validator("class_size", mode="before")
    @classmethod
    def _norm_class_size(cls, v: object) -> str:
        return parse_enum(ClassSizeEnum, str(v), "class size").value

    @field_validator("teaching_subjects", mode="before")
    @classmethod
    def _norm_subjects(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("teaching_subjects must be a list")
        return [s.value for s in parse_enum_list(TeachingSubjectEnum, [str(x) for x in v], "subject")]

    @field_validator("teaching_modes", mode="before")
    @classmethod
    def _norm_modes(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError("teaching_modes must be a list")
        return [m.value for m in parse_enum_list(TeachingModeEnum, [str(x) for x in v], "teaching mode")]

    @field_validator("bio", mode="before")
    @classmethod
    def _strip_bio(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class SchoolSignupRequest(BaseModel):
    school_name: str = Field(min_length=2, max_length=255)
    school_email: EmailStr
    phone: str = Field(min_length=8, max_length=50)
    address: str = Field(min_length=2, max_length=500)
    full_name: str = Field(min_length=2, max_length=255, description="Administrator name")
    designation: str = Field(min_length=2, max_length=100)
    recovery_email: EmailStr
    mobile: str = Field(min_length=8, max_length=50)
    user_id: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    grades_offered: str = Field(min_length=1, max_length=80)
    student_strength: str = Field(min_length=1, max_length=50)
    curricula: list[str] = Field(min_length=1, max_length=6)
    website: str | None = Field(default=None, max_length=500)

    @field_validator("school_name", "full_name", "designation", "address", mode="before")
    @classmethod
    def _strip_required(cls, v: object) -> str:
        s = str(v).strip()
        if len(s) < 2:
            raise ValueError("This field is required.")
        return s

    @field_validator("user_id", mode="before")
    @classmethod
    def _norm_user_id(cls, v: object) -> str:
        u = normalize_username(str(v))
        if not u:
            raise ValueError("User ID is required.")
        if not u.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                "User ID can only use letters, numbers, hyphens, underscores, and dots."
            )
        return u

    @field_validator("phone", "mobile", mode="before")
    @classmethod
    def _norm_phone(cls, v: object) -> str:
        p = normalize_phone(str(v))
        if not p or len(p) < 8:
            raise ValueError("Enter a valid phone number.")
        return p

    @field_validator("grades_offered", mode="before")
    @classmethod
    def _norm_grades_offered(cls, v: object) -> str:
        return parse_enum(SchoolGradeRangeEnum, str(v), "grades offered").value

    @field_validator("student_strength", mode="before")
    @classmethod
    def _norm_strength(cls, v: object) -> str:
        return parse_enum(StudentStrengthEnum, str(v), "student strength").value

    @field_validator("curricula", mode="before")
    @classmethod
    def _norm_curricula(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("curricula must be a list")
        return [c.value for c in parse_curricula([str(x) for x in v])]

    @field_validator("website", mode="before")
    @classmethod
    def _strip_website(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

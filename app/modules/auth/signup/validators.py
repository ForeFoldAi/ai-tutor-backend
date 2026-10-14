"""Shared signup field normalization and validation helpers."""

from __future__ import annotations

import re

from fastapi import status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.student_messages import EMAIL_ALREADY_USED, USERNAME_TAKEN
from app.modules.auth.exceptions import AuthException
from app.modules.auth.signup.enums import (
    ClassSizeEnum,
    CurriculumEnum,
    LearningGoalEnum,
    LearningMethodEnum,
    SchoolGradeRangeEnum,
    StudentGradeEnum,
    StudentStrengthEnum,
    StudentSubjectEnum,
    TeachingExperienceEnum,
    TeachingModeEnum,
    TeachingSubjectEnum,
    TutorGradeRangeEnum,
    parse_enum,
    parse_enum_list,
)
from app.modules.users.models import User, UserSettings

_PHONE_DIGITS = re.compile(r"\D+")


def normalize_phone(phone: str | None) -> str | None:
    """Strip to digits for lookup/storage; ponytail: no country-code normalization."""
    if phone is None:
        return None
    digits = _PHONE_DIGITS.sub("", phone.strip())
    return digits if digits else None


def normalize_username(username: str | None) -> str | None:
    if username is None:
        return None
    s = username.strip().lower().replace(" ", "")
    return s if s else None


def normalize_grade(grade: str) -> str:
    """Accept 'Grade 6', '6', etc. → canonical grade digit string."""
    s = str(grade).strip()
    if s.lower().startswith("grade "):
        s = s[6:].strip()
    return s


def parse_student_grade(grade: str) -> StudentGradeEnum:
    g = normalize_grade(grade)
    return parse_enum(StudentGradeEnum, g, "grade")  # type: ignore[return-value]


def parse_curricula(values: list[str]) -> list[CurriculumEnum]:
    return parse_enum_list(CurriculumEnum, values, "curriculum")  # type: ignore[return-value]


def parse_student_subjects(values: list[str]) -> list[StudentSubjectEnum]:
    if not values:
        return []
    return parse_enum_list(StudentSubjectEnum, values, "subject")  # type: ignore[return-value]


def parse_learning_goals(values: list[str]) -> list[LearningGoalEnum]:
    if not values:
        return []
    return parse_enum_list(LearningGoalEnum, values, "learning goal")  # type: ignore[return-value]


def parse_learning_method(value: str | None) -> LearningMethodEnum | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return parse_enum(LearningMethodEnum, s, "learning method")  # type: ignore[return-value]


def find_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


def find_user_by_phone(db: Session, phone: str) -> User | None:
    normalized = normalize_phone(phone)
    if not normalized:
        return None
    return db.scalar(select(User).where(User.phone == normalized))


def find_user_by_username(db: Session, username: str) -> User | None:
    ident = normalize_username(username)
    if not ident:
        return None
    settings = db.scalar(select(UserSettings).where(func.lower(UserSettings.username) == ident))
    if not settings:
        return None
    return db.get(User, settings.user_id)


def ensure_email_available(db: Session, email: str) -> None:
    if find_user_by_email(db, email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)


def ensure_phone_available(
    db: Session,
    phone: str | None,
    *,
    exclude_user_id: int | None = None,
) -> None:
    if not phone:
        return
    existing = find_user_by_phone(db, phone)
    if existing and existing.id != exclude_user_id:
        raise AuthException("This phone number is already registered.", status.HTTP_409_CONFLICT)


def ensure_username_available(db: Session, username: str | None) -> None:
    if not username:
        return
    if find_user_by_username(db, username):
        raise AuthException(USERNAME_TAKEN, status.HTTP_409_CONFLICT)

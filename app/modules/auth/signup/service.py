"""Public signup business logic for student, tutor, and school accounts."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.modules.auth.constants import Role
from app.modules.auth.service import get_or_create_user_settings
from app.modules.auth.signup.enums import (
    ClassSizeEnum,
    SchoolGradeRangeEnum,
    StudentStrengthEnum,
    TeachingExperienceEnum,
    TeachingModeEnum,
    TeachingSubjectEnum,
    TutorGradeRangeEnum,
    parse_enum,
    parse_enum_list,
)
from app.modules.auth.signup.models import UserSignupProfile
from app.modules.auth.signup.schemas import (
    SchoolSignupRequest,
    StudentSignupRequest,
    TutorSignupRequest,
)
from app.modules.auth.signup.validators import (
    ensure_username_available,
    parse_curricula,
    parse_learning_goals,
    parse_learning_method,
    parse_student_grade,
    parse_student_subjects,
)
from app.modules.schools.models import School
from app.modules.users.models import User


def _assign_username(db: Session, user: User, username: str) -> None:
    settings = get_or_create_user_settings(db, user)
    settings.username = username.lower()


def signup_student(db: Session, payload: StudentSignupRequest) -> User:
    ensure_username_available(db, payload.user_id)

    curricula = parse_curricula(payload.curricula)
    grade = parse_student_grade(payload.grade)
    subjects = parse_student_subjects(payload.favorite_subjects)
    goals = parse_learning_goals(payload.learning_goals)
    method = parse_learning_method(payload.preferred_learning_method)

    user = User(
        full_name=payload.full_name,
        email=str(payload.email).lower(),
        password_hash=hash_password(payload.password),
        role=Role.STUDENT,
        is_active=True,
        is_verified=False,
        phone=payload.parent_phone,
        teaching_board=curricula[0].value,
        teaching_classes=[{"grade": grade.value, "sections": ["A"]}],
    )
    db.add(user)
    db.flush()

    db.add(
        UserSignupProfile(
            user_id=user.id,
            student_grade=grade,
            curricula=curricula,
            favorite_subjects=subjects or None,
            learning_goals=goals or None,
            preferred_learning_method=method,
            parent_email=str(payload.parent_email) if payload.parent_email else None,
        )
    )
    db.flush()
    _assign_username(db, user, payload.user_id)
    return user


def signup_tutor(db: Session, payload: TutorSignupRequest) -> User:
    ensure_username_available(db, payload.user_id)

    subjects = parse_enum_list(TeachingSubjectEnum, payload.teaching_subjects, "subject")
    user = User(
        full_name=payload.full_name,
        email=str(payload.email).lower(),
        password_hash=hash_password(payload.password),
        role=Role.TUTOR,
        is_active=True,
        is_verified=False,
        phone=payload.mobile,
        teaching_subjects=[e.value if hasattr(e, "value") else str(e) for e in subjects],
    )
    db.add(user)
    db.flush()

    db.add(
        UserSignupProfile(
            user_id=user.id,
            teaching_experience=parse_enum(TeachingExperienceEnum, payload.teaching_experience, "experience"),
            grades_teach=parse_enum(TutorGradeRangeEnum, payload.grades_teach, "grade range"),
            class_size=parse_enum(ClassSizeEnum, payload.class_size, "class size"),
            teaching_subjects=subjects,
            teaching_modes=parse_enum_list(TeachingModeEnum, payload.teaching_modes, "teaching mode"),
            bio=payload.bio,
        )
    )
    db.flush()
    _assign_username(db, user, payload.user_id)
    return user


def signup_school(db: Session, payload: SchoolSignupRequest) -> User:
    """Register a school workspace: school record + school admin."""
    ensure_username_available(db, payload.user_id)

    curricula = parse_curricula(payload.curricula)
    grades_offered = parse_enum(SchoolGradeRangeEnum, payload.grades_offered, "grades offered")
    student_strength = parse_enum(StudentStrengthEnum, payload.student_strength, "student strength")

    school = School(
        name=payload.school_name,
        email=str(payload.school_email).lower(),
        phone=payload.phone,
        address=payload.address,
        website=payload.website,
        grades_offered=grades_offered,
        student_strength=student_strength,
        curricula=curricula,
        board=curricula[0].value,
        is_active=True,
    )
    db.add(school)
    db.flush()

    user = User(
        full_name=payload.full_name,
        email=str(payload.recovery_email).lower(),
        password_hash=hash_password(payload.password),
        role=Role.SCHOOL_ADMIN,
        is_active=True,
        is_verified=False,
        school_id=school.id,
        phone=payload.mobile,
    )
    db.add(user)
    db.flush()

    db.add(UserSignupProfile(user_id=user.id, designation=payload.designation))
    db.flush()
    _assign_username(db, user, payload.user_id)
    return user


def signup_success_message(user: User, verify_token: str) -> str:
    return (
        f"{user.role.value} signup successful. "
        f"Verification token generated (hook): {verify_token[:16]}..."
    )

"""Public sequential ids — after UUID→bigint migration, model PKs ARE sequential.

Keep thin helpers so call sites stay readable; no UUID remapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.auth.exceptions import AuthException
from app.modules.schools.models import School
from app.modules.users.models import User

if TYPE_CHECKING:
    from app.modules.auth.schemas import (
        SchoolAdminBrief,
        SchoolDetailResponse,
        UserResponse,
        UserSettingsResponse,
    )


def school_by_seq(db: Session, seq: int) -> School | None:
    """Lookup school by public id (PK)."""
    return db.get(School, seq)


def school_seq(db: Session, school_id: int | None) -> int | None:
    if school_id is None:
        return None
    school = db.get(School, school_id)
    return school.id if school else None


def user_by_account_id(db: Session, account_id: int) -> User | None:
    """Lookup user by public id (PK). Name kept for call-site compatibility."""
    return db.get(User, account_id)


def require_user_by_account_id(db: Session, account_id: int) -> User:
    user = user_by_account_id(db, account_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    return user


def require_school_by_seq(db: Session, seq: int) -> School:
    school = school_by_seq(db, seq)
    if not school:
        raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
    return school


def resolve_school_uuid(db: Session, school_ref: int | None) -> int | None:
    """Resolve public school id (historically named resolve_school_uuid)."""
    if school_ref is None:
        return None
    return require_school_by_seq(db, int(school_ref)).id


def coerce_teaching_classes(raw: Any) -> list[dict] | None:
    if raw is None:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return [{"grade": str(g), "sections": ["A"]} for g in raw if str(g).strip()]
    if isinstance(raw, list):
        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            grade = str(item.get("grade", "")).strip()
            sections = [
                str(s).strip().upper()
                for s in (item.get("sections") or [])
                if str(s).strip()
            ]
            if not (grade and sections):
                continue
            row: dict = {"grade": grade, "sections": sections}
            curriculum = str(item.get("curriculum") or "").strip()
            if curriculum:
                row["curriculum"] = curriculum
            sc_id = item.get("school_class_id")
            if sc_id is not None and str(sc_id).strip().lstrip("-").isdigit():
                row["school_class_id"] = int(sc_id)
            out.append(row)
        return out
    return None


def _enrich_teaching_classes_curriculum(db: Session, user: User, rows: list[dict]) -> list[dict]:
    """Fill missing curriculum from school_classes (by seq or grade+section)."""
    if not rows or not user.school_id:
        return rows

    missing = [r for r in rows if not r.get("curriculum")]
    if not missing:
        return rows

    from app.modules.school_admin.classes.models import SchoolClass

    school_classes = list(
        db.scalars(select(SchoolClass).where(SchoolClass.school_id == user.school_id))
    )
    by_seq = {c.seq: c for c in school_classes}
    by_grade_section = {
        (c.grade.strip(), c.section.strip().upper()): c for c in school_classes
    }

    for row in rows:
        if row.get("curriculum"):
            continue
        sc = None
        sc_id = row.get("school_class_id")
        if sc_id is not None:
            sc = by_seq.get(int(sc_id))
        if sc is None:
            grade = str(row.get("grade", "")).strip()
            for section in row.get("sections") or []:
                sc = by_grade_section.get((grade, str(section).strip().upper()))
                if sc:
                    break
        if sc is None:
            continue
        row["curriculum"] = sc.curriculum
        row["school_class_id"] = sc.seq
    return rows


def _curriculum_from_classes(raw: Any) -> str | None:
    if not isinstance(raw, list):
        return None
    for item in raw:
        if isinstance(item, dict):
            cur = str(item.get("curriculum") or "").strip()
            if cur:
                return cur
    return None


def _effective_teaching_board(user: User, teaching_classes: list[dict] | None = None) -> str | None:
    """Prefer a real curriculum/board; never surface a subject name as board."""
    from app.modules.auth.signup.enums import CurriculumEnum, TeachingSubjectEnum

    board = (user.teaching_board or "").strip() or None
    class_cur = _curriculum_from_classes(
        teaching_classes if teaching_classes is not None else user.teaching_classes
    )
    known_boards = {m.value.lower() for m in CurriculumEnum}
    subject_names = {m.value.lower() for m in TeachingSubjectEnum}

    if board and board.lower() in known_boards:
        return board
    if class_cur:
        return class_cur
    if board and board.lower() in subject_names:
        return None
    return board


def heal_tutor_teaching_curriculum(db: Session, user: User) -> bool:
    """Fix school tutors whose teaching_board was overwritten with a subject name."""
    from app.modules.auth.constants import Role
    from app.modules.auth.signup.enums import TeachingSubjectEnum

    if user.role != Role.TUTOR or not user.school_id:
        return False

    teaching_raw = coerce_teaching_classes(user.teaching_classes) or []
    teaching_raw = _enrich_teaching_classes_curriculum(db, user, teaching_raw)
    board = _effective_teaching_board(user, teaching_raw)
    if not board and user.school_id:
        school = db.get(School, user.school_id)
        if school and school.board and str(school.board).strip():
            board = str(school.board).strip()

    subject_names = {m.value.lower() for m in TeachingSubjectEnum}
    board_raw = (user.teaching_board or "").strip()
    raw_classes = user.teaching_classes if isinstance(user.teaching_classes, list) else []
    needs_class_heal = bool(teaching_raw) and any(
        isinstance(item, dict) and not str(item.get("curriculum") or "").strip() for item in raw_classes
    )
    needs_board_heal = bool(board) and (not board_raw or board_raw.lower() in subject_names)
    if not (needs_class_heal or needs_board_heal):
        return False
    if needs_class_heal:
        user.teaching_classes = teaching_raw
    if needs_board_heal:
        user.teaching_board = board
    db.add(user)
    return True


def to_user_response(db: Session, user: User) -> "UserResponse":
    from app.modules.auth.constants import Role
    from app.modules.auth.schemas import TeachingClassAssignment, UserResponse
    from app.modules.auth.signup.models import UserSignupProfile

    created_by = None
    if user.created_by:
        creator = db.get(User, user.created_by)
        created_by = creator.id if creator else None

    teaching_raw = coerce_teaching_classes(user.teaching_classes) or []
    teaching_raw = _enrich_teaching_classes_curriculum(db, user, teaching_raw)
    board = _effective_teaching_board(user, teaching_raw)

    if not board and user.school_id:
        school = db.get(School, user.school_id)
        if school and school.board and str(school.board).strip():
            board = str(school.board).strip()

    teaching_classes = (
        [TeachingClassAssignment.model_validate(row) for row in teaching_raw] if teaching_raw else None
    )
    signup_profile = db.get(UserSignupProfile, user.id)
    student_grade = None
    curricula: list[str] | None = None
    parent_email = None
    favorite_subjects: list[str] | None = None
    learning_goals: list[str] | None = None
    preferred_learning_method = None
    # Signup personal details are for individual (untagged) students only.
    is_individual = user.school_id is None and user.created_by is None
    if signup_profile and user.role == Role.STUDENT and is_individual:
        if signup_profile.student_grade is not None:
            student_grade = f"Grade {signup_profile.student_grade.value}"
        if signup_profile.curricula:
            curricula = [c.value for c in signup_profile.curricula]
        parent_email = signup_profile.parent_email
        if signup_profile.favorite_subjects:
            favorite_subjects = [s.value for s in signup_profile.favorite_subjects]
        if signup_profile.learning_goals:
            learning_goals = [g.value for g in signup_profile.learning_goals]
        if signup_profile.preferred_learning_method is not None:
            preferred_learning_method = signup_profile.preferred_learning_method.value

    return UserResponse(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        school_id=user.school_id,
        phone=user.phone,
        designation=signup_profile.designation if signup_profile else None,
        teaching_board=board,
        teaching_subjects=list(user.teaching_subjects) if user.teaching_subjects else None,
        teaching_classes=teaching_classes,
        student_grade=student_grade,
        curricula=curricula,
        parent_email=parent_email,
        favorite_subjects=favorite_subjects,
        learning_goals=learning_goals,
        preferred_learning_method=preferred_learning_method,
        created_by=created_by,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def to_user_settings_response(user: User, settings: Any) -> "UserSettingsResponse":
    from app.modules.auth.schemas import UserSettingsResponse

    return UserSettingsResponse(
        user_id=user.id,
        username=settings.username,
        language=settings.language,
        theme=settings.theme,
        notify_email=settings.notify_email,
        notify_push=settings.notify_push,
        notify_assignments=settings.notify_assignments,
        notify_sessions=settings.notify_sessions,
        notify_messages=settings.notify_messages,
        created_at=settings.created_at,
        updated_at=settings.updated_at,
    )


def to_school_admin_brief(user: User) -> "SchoolAdminBrief":
    from app.modules.auth.schemas import SchoolAdminBrief

    return SchoolAdminBrief(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        is_active=user.is_active,
    )


def to_school_detail(school: School) -> "SchoolDetailResponse":
    from app.modules.auth.schemas import SchoolDetailResponse

    curricula: list[str] = []
    if school.curricula:
        curricula = [c.value if hasattr(c, "value") else str(c) for c in school.curricula]
    grades = school.grades_offered.value if school.grades_offered is not None else None
    strength = school.student_strength.value if school.student_strength is not None else None

    return SchoolDetailResponse(
        id=school.id,
        name=school.name,
        branch=school.branch,
        board=school.board,
        email=school.email,
        phone=school.phone,
        website=school.website,
        address=school.address,
        grades_offered=grades,
        student_strength=strength,
        curricula=curricula,
        is_active=school.is_active,
        created_at=school.created_at,
    )

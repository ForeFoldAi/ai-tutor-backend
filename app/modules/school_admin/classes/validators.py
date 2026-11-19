from __future__ import annotations

from dataclasses import dataclass

from fastapi import status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import ColumnElement

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.school_admin.classes.models import SchoolClass, SchoolSubject
from app.modules.schools.models import School
from app.modules.users.models import User

# Catalog boards when individual tutors have no school curricula list.
DEFAULT_CURRICULA = ["CBSE", "ICSE", "STATE_BOARD", "IB", "CAMBRIDGE"]


def is_individual_tutor(actor: User) -> bool:
    return actor.role == Role.TUTOR and actor.school_id is None


def assert_can_manage_classes(actor: User) -> None:
    """School admin / master admin, or individual tutor (no school)."""
    if actor.role in (Role.SCHOOL_ADMIN, Role.MASTER_ADMIN):
        return
    if is_individual_tutor(actor):
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


@dataclass(frozen=True)
class ClassScope:
    school_id: int | None
    owner_user_id: int | None

    def class_filter(self) -> ColumnElement[bool]:
        if self.school_id is not None:
            return SchoolClass.school_id == self.school_id
        return (SchoolClass.owner_user_id == self.owner_user_id) & SchoolClass.school_id.is_(None)

    def subject_filter(self) -> ColumnElement[bool]:
        if self.school_id is not None:
            return SchoolSubject.school_id == self.school_id
        return (SchoolSubject.owner_user_id == self.owner_user_id) & SchoolSubject.school_id.is_(None)


def resolve_class_scope(
    actor: User,
    *,
    school_id: int | None = None,
) -> ClassScope:
    if is_individual_tutor(actor):
        return ClassScope(school_id=None, owner_user_id=actor.id)
    if actor.role == Role.SCHOOL_ADMIN:
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        return ClassScope(school_id=actor.school_id, owner_user_id=None)
    if actor.role == Role.MASTER_ADMIN:
        if school_id is None:
            raise AuthException("school_id is required.", status.HTTP_400_BAD_REQUEST)
        return ClassScope(school_id=school_id, owner_user_id=None)
    # School-tagged tutors: read-only school scope for move-class dialogs.
    if actor.role == Role.TUTOR and actor.school_id:
        return ClassScope(school_id=actor.school_id, owner_user_id=None)
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def resolve_target_school_id(
    actor: User,
    *,
    school_id: int | None = None,
) -> int:
    """School-scoped actors only. Individual tutors must use resolve_class_scope."""
    if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR):
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        return actor.school_id
    if actor.role == Role.MASTER_ADMIN:
        if school_id is None:
            raise AuthException("school_id is required.", status.HTTP_400_BAD_REQUEST)
        return school_id
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def get_actor_school(db: Session, actor: User, *, school_id: int | None = None) -> School:
    target_id = resolve_target_school_id(actor, school_id=school_id)
    school = db.get(School, target_id)
    if not school:
        raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
    return school


def school_curricula(school: School) -> list[str]:
    if school.curricula:
        return [c.value if hasattr(c, "value") else str(c) for c in school.curricula]
    if school.board:
        return [school.board]
    return []


def scope_curricula(db: Session, actor: User, scope: ClassScope) -> list[str]:
    if scope.school_id is not None:
        school = db.get(School, scope.school_id)
        if school:
            return school_curricula(school) or list(DEFAULT_CURRICULA)
        return list(DEFAULT_CURRICULA)
    if actor.teaching_board and actor.teaching_board.strip():
        board = actor.teaching_board.strip()
        extras = [c for c in DEFAULT_CURRICULA if c.lower() != board.lower()]
        return [board, *extras]
    return list(DEFAULT_CURRICULA)


def assert_curriculum_allowed(school: School, curriculum: str) -> str:
    value = curriculum.strip()
    if not value:
        raise AuthException("Curriculum is required.", status.HTTP_400_BAD_REQUEST)
    allowed = {c.lower() for c in school_curricula(school)}
    if allowed and value.lower() not in allowed:
        raise AuthException(
            f"Curriculum '{value}' is not offered by this school.",
            status.HTTP_400_BAD_REQUEST,
        )
    return value


def assert_curriculum_in_scope(allowed: list[str], curriculum: str) -> str:
    value = curriculum.strip()
    if not value:
        raise AuthException("Curriculum is required.", status.HTTP_400_BAD_REQUEST)
    if allowed and value.lower() not in {c.lower() for c in allowed}:
        raise AuthException(
            f"Curriculum '{value}' is not available.",
            status.HTTP_400_BAD_REQUEST,
        )
    return value


def assert_actor_manages_class(actor: User, school_class: SchoolClass) -> None:
    if actor.role == Role.MASTER_ADMIN:
        return
    if actor.role == Role.SCHOOL_ADMIN:
        if school_class.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    if is_individual_tutor(actor):
        if school_class.owner_user_id != actor.id or school_class.school_id is not None:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def assert_actor_manages_subject(actor: User, subject: SchoolSubject) -> None:
    if actor.role == Role.MASTER_ADMIN:
        return
    if actor.role == Role.SCHOOL_ADMIN:
        if subject.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    if is_individual_tutor(actor):
        if subject.owner_user_id != actor.id or subject.school_id is not None:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def sync_individual_tutor_teaching_profile(db: Session, tutor: User) -> None:
    """Mirror owned classes/subjects onto User fields used by Sessions / Lesson Planner."""
    if not is_individual_tutor(tutor):
        return
    classes = list(
        db.scalars(
            select(SchoolClass)
            .where(SchoolClass.owner_user_id == tutor.id, SchoolClass.school_id.is_(None))
            .order_by(SchoolClass.grade, SchoolClass.section)
        )
    )
    subjects = list(
        db.scalars(
            select(SchoolSubject).where(
                SchoolSubject.owner_user_id == tutor.id,
                SchoolSubject.school_id.is_(None),
                SchoolSubject.is_active.is_(True),
            )
        )
    )
    by_grade: dict[str, dict] = {}
    for row in classes:
        entry = by_grade.setdefault(
            row.grade,
            {
                "grade": row.grade,
                "sections": [],
                "curriculum": row.curriculum,
                "school_class_id": row.seq,
            },
        )
        sec = row.section.strip().upper()
        if sec and sec not in entry["sections"]:
            entry["sections"].append(sec)
        entry["curriculum"] = row.curriculum
        entry["school_class_id"] = row.seq
    tutor.teaching_classes = list(by_grade.values()) or None
    if classes:
        tutor.teaching_board = classes[0].curriculum
    tutor.teaching_subjects = [s.name for s in subjects] or tutor.teaching_subjects

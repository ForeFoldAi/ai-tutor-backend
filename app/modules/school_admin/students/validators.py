from __future__ import annotations

from fastapi import status

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.school_admin.classes.validators import is_individual_tutor
from app.modules.users.models import User

__all__ = [
    "assert_actor_manages_student",
    "is_individual_tutor",
    "resolve_target_school_id",
]


def resolve_target_school_id(
    actor: User,
    *,
    school_id: int | None = None,
) -> int:
    if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR):
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        return actor.school_id
    if actor.role == Role.MASTER_ADMIN:
        if school_id is None:
            raise AuthException("school_id is required.", status.HTTP_400_BAD_REQUEST)
        return school_id
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def assert_actor_manages_student(actor: User, student: User) -> None:
    if student.role != Role.STUDENT:
        raise AuthException("Not a student account.", status.HTTP_400_BAD_REQUEST)
    if actor.role == Role.MASTER_ADMIN:
        return
    if is_individual_tutor(actor):
        if student.created_by == actor.id and student.school_id is None:
            return
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR):
        if not actor.school_id or student.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


if __name__ == "__main__":
    # ponytail: ownership guard for individual tutors (no school).
    Fake = type("U", (), {})
    actor = Fake()
    actor.id = 7
    actor.role = Role.TUTOR
    actor.school_id = None
    owned = Fake()
    owned.role = Role.STUDENT
    owned.created_by = 7
    owned.school_id = None
    assert_actor_manages_student(actor, owned)
    other = Fake()
    other.role = Role.STUDENT
    other.created_by = 99
    other.school_id = None
    try:
        assert_actor_manages_student(actor, other)
        raise SystemExit("expected Forbidden")
    except AuthException:
        pass
    print("students validators self-check ok")

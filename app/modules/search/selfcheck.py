"""ponytail: assert-based self-check for role→entity map + hit shaping (no DB)."""

from __future__ import annotations

from app.modules.auth.constants import Role
from app.modules.search.constants import (
    ENTITY_LESSONS,
    ENTITY_SESSIONS,
    ENTITY_SUBJECTS,
    ENTITY_STUDENTS,
    ENTITY_TEACHERS,
    ROLE_GLOBAL_ENTITIES,
)
from app.modules.search.service import _hit, _href_for, _items_to_hits


class _Actor:
    def __init__(self, role: Role, school_id: int | None = 1):
        self.role = role
        self.school_id = school_id


class _Student:
    id = 7
    full_name = "Ada Lovelace"
    email = "ada@example.com"
    user_id = "ada01"


def main() -> None:
    assert ENTITY_STUDENTS in ROLE_GLOBAL_ENTITIES[Role.SCHOOL_ADMIN]
    assert ENTITY_TEACHERS in ROLE_GLOBAL_ENTITIES[Role.SCHOOL_ADMIN]
    assert ENTITY_STUDENTS in ROLE_GLOBAL_ENTITIES[Role.TUTOR]
    assert ENTITY_SESSIONS in ROLE_GLOBAL_ENTITIES[Role.TUTOR]
    assert ROLE_GLOBAL_ENTITIES[Role.STUDENT] == (
        ENTITY_SUBJECTS,
        ENTITY_LESSONS,
        ENTITY_SESSIONS,
    )

    tutor = _Actor(Role.TUTOR)
    assert _href_for(tutor, ENTITY_STUDENTS) == "/tutor/students"
    student = _Actor(Role.STUDENT)
    assert _href_for(student, ENTITY_SESSIONS) == "/live-classes"

    hit = _hit("students", 1, "Ada", "ada@x.com", "/students")
    assert hit.id == "1" and hit.title == "Ada"

    hits = _items_to_hits(_Actor(Role.SCHOOL_ADMIN), ENTITY_STUDENTS, [_Student()])
    assert len(hits) == 1
    assert hits[0].title == "Ada Lovelace"
    assert hits[0].href == "/students"
    print("search self-check ok")


if __name__ == "__main__":
    main()
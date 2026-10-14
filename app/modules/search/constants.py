from __future__ import annotations

from app.modules.auth.constants import Role

# Entity keys used by /search/global and /search/{entity}
ENTITY_STUDENTS = "students"
ENTITY_TEACHERS = "teachers"
ENTITY_CLASSES = "classes"
ENTITY_CREDENTIALS = "credentials"
ENTITY_SESSIONS = "sessions"
ENTITY_SUBJECTS = "subjects"
ENTITY_LESSONS = "lessons"
ENTITY_USERS = "users"
ENTITY_SCHOOLS = "schools"

ALL_ENTITIES = frozenset(
    {
        ENTITY_STUDENTS,
        ENTITY_TEACHERS,
        ENTITY_CLASSES,
        ENTITY_CREDENTIALS,
        ENTITY_SESSIONS,
        ENTITY_SUBJECTS,
        ENTITY_LESSONS,
        ENTITY_USERS,
        ENTITY_SCHOOLS,
    }
)

# Dashboard global search: which entities each role can query.
ROLE_GLOBAL_ENTITIES: dict[Role, tuple[str, ...]] = {
    Role.SCHOOL_ADMIN: (
        ENTITY_STUDENTS,
        ENTITY_TEACHERS,
        ENTITY_CLASSES,
        ENTITY_CREDENTIALS,
    ),
    Role.TUTOR: (ENTITY_STUDENTS, ENTITY_SESSIONS),
    Role.MASTER_ADMIN: (ENTITY_USERS, ENTITY_SCHOOLS),
    Role.STUDENT: (ENTITY_SUBJECTS, ENTITY_LESSONS, ENTITY_SESSIONS),
}

# Tab entity search: roles allowed to hit each entity endpoint.
ENTITY_ROLES: dict[str, frozenset[Role]] = {
    ENTITY_STUDENTS: frozenset({Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR}),
    ENTITY_TEACHERS: frozenset({Role.SCHOOL_ADMIN, Role.MASTER_ADMIN}),
    ENTITY_CLASSES: frozenset({Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR}),
    ENTITY_CREDENTIALS: frozenset({Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR}),
    ENTITY_SESSIONS: frozenset({Role.TUTOR, Role.STUDENT}),
    ENTITY_SUBJECTS: frozenset({Role.STUDENT}),
    ENTITY_LESSONS: frozenset({Role.STUDENT}),
    ENTITY_USERS: frozenset({Role.MASTER_ADMIN, Role.SCHOOL_ADMIN}),
    ENTITY_SCHOOLS: frozenset({Role.MASTER_ADMIN}),
}

# Frontend hrefs for global-hit navigation (role-specific overrides in service).
ENTITY_HREF: dict[str, str] = {
    ENTITY_STUDENTS: "/students",
    ENTITY_TEACHERS: "/teachers",
    ENTITY_CLASSES: "/classes",
    ENTITY_CREDENTIALS: "/credentials",
    ENTITY_SESSIONS: "/tutor/sessions",
    ENTITY_SUBJECTS: "/ai-learning-studio",
    ENTITY_LESSONS: "/my-learning",
    ENTITY_USERS: "/master-admin/users",
    ENTITY_SCHOOLS: "/master-admin/schools",
}

# ponytail: ceil at list-module MAX — upgrade to unlimited cursor scan if schools exceed this.
GLOBAL_PER_ENTITY_LIMIT = 8

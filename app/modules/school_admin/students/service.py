from __future__ import annotations

import csv
import io
import secrets
from datetime import UTC, datetime

from fastapi import status
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.public_ids import require_user_by_account_id, school_seq
from app.modules.auth.service import get_or_create_user_settings, list_tutor_assigned_students
from app.modules.auth.signup.validators import (
    ensure_username_available,
    normalize_phone,
    normalize_username,
)
from app.modules.school_admin.classes.models import SchoolClass
from app.modules.notifications.service import notify_users
from app.modules.student_learning.enrollment import (
    enrolled_subject_names,
    resolve_student_school_class,
)
from app.modules.school_admin.students.constants import (
    DEFAULT_PAGE_LIMIT,
    LEARNING_TYPES,
    MAX_BULK_IMPORT,
    MAX_PAGE_LIMIT,
    PAGE_SIZE_OPTIONS,
)
from app.modules.school_admin.students.schemas import (
    BulkRowError,
    PaginationMeta,
    StudentActionResponse,
    StudentAssignTeacherRequest,
    StudentBulkCreateResponse,
    StudentCreateItem,
    StudentListResponse,
    StudentMoveClassRequest,
    StudentOptionsResponse,
    StudentResponse,
    StudentUpdateRequest,
)
from app.modules.school_admin.students.validators import (
    assert_actor_manages_student,
    is_individual_tutor,
    resolve_target_school_id,
)
from app.modules.school_admin.teachers.validators import username_taken
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User, UserSettings


PASSWORD_NOT_SET = "Not Set"
PASSWORD_GENERATED = "Generated"
PASSWORD_LOGGED_IN = "Logged In"


def _username_for_user(db: Session, user_id: int) -> str:
    row = db.get(UserSettings, user_id)
    if row and row.username:
        return row.username
    user = db.get(User, user_id)
    if user:
        settings = get_or_create_user_settings(db, user)
        return settings.username or user.email.split("@", 1)[0]
    return ""


def _class_fields(student: User) -> tuple[str | None, str | None, str | None]:
    raw = student.teaching_classes
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        grade = str(raw[0].get("grade", "")).strip() or None
        secs = raw[0].get("sections") or []
        section = str(secs[0]).strip().upper() if secs else None
        curriculum = str(raw[0].get("curriculum", "")).strip() or None
        if not curriculum and student.teaching_board:
            curriculum = student.teaching_board.strip() or None
        return grade, section, curriculum
    board = student.teaching_board.strip() if student.teaching_board else None
    return None, None, board


def _normalized_student_row(student: User) -> tuple[str, set[str]] | None:
    grade, section, _ = _class_fields(student)
    if not grade or not section:
        return None
    return grade, {section}


def _tutors_for_student(db: Session, student: User) -> list[User]:
    if not student.school_id:
        return []
    student_row = _normalized_student_row(student)
    if not student_row:
        return []
    s_grade, s_sections = student_row
    tutors = list(
        db.scalars(
            select(User).where(
                User.role == Role.TUTOR,
                User.school_id == student.school_id,
                User.is_active.is_(True),
            )
        )
    )
    matched: list[User] = []
    for tutor in tutors:
        excluded = {
            int(x)
            for x in (tutor.excluded_student_ids or [])
            if str(x).strip().lstrip("-").isdigit()
        }
        if student.id in excluded:
            continue
        raw = tutor.teaching_classes
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            t_grade = str(item.get("grade", "")).strip()
            t_sections = {
                str(s).strip().upper() for s in (item.get("sections") or []) if str(s).strip()
            }
            if t_grade == s_grade and t_sections.intersection(s_sections):
                matched.append(tutor)
                break
    return matched


def _learning_fields_from_tutors(tutors: list[User]) -> tuple[str, str | None, list[int]]:
    if tutors:
        return (
            "Teacher Guided",
            ", ".join(t.full_name for t in tutors),
            [t.id for t in tutors],
        )
    return "Self Learning", None, []


def _subject_learning_split(
    db: Session,
    student: User,
    *,
    tutors: list[User] | None = None,
) -> tuple[list[str], list[str]]:
    """Split class subjects into (teacher_guided, self_learned)."""
    school_class = resolve_student_school_class(db, student)
    if school_class is None:
        return [], []
    class_subjects = enrolled_subject_names(db, school_class)
    if not class_subjects:
        return [], []
    matched = tutors if tutors is not None else _tutors_for_student(db, student)
    covered: set[str] = set()
    for tutor in matched:
        for name in tutor.teaching_subjects or []:
            key = str(name).strip().lower()
            if key:
                covered.add(key)
    guided = [n for n in class_subjects if n.strip().lower() in covered]
    self_learned = [n for n in class_subjects if n.strip().lower() not in covered]
    return guided, self_learned


def _password_status(student: User, *, has_logged_in: bool) -> str:
    """Not Set → Generated (creds issued) → Logged In (first session)."""
    if has_logged_in:
        return PASSWORD_LOGGED_IN
    if student.credentials_generated_at is not None:
        return PASSWORD_GENERATED
    return PASSWORD_NOT_SET


def _logged_in_user_ids(db: Session, user_ids: list[int]) -> set[int]:
    if not user_ids:
        return set()
    rows = db.scalars(
        select(SessionToken.user_id).where(SessionToken.user_id.in_(user_ids)).distinct()
    ).all()
    return set(rows)


def _student_response(
    db: Session,
    student: User,
    *,
    has_logged_in: bool | None = None,
) -> StudentResponse:
    grade, section, curriculum = _class_fields(student)
    tutors = _tutors_for_student(db, student)
    learning_type, learning_teacher, learning_teacher_ids = _learning_fields_from_tutors(tutors)
    guided, self_learned = _subject_learning_split(db, student, tutors=tutors)
    if has_logged_in is None:
        has_logged_in = bool(
            db.scalar(select(SessionToken.id).where(SessionToken.user_id == student.id).limit(1))
        )
    return StudentResponse(
        id=student.id,
        user_id=_username_for_user(db, student.id),
        full_name=student.full_name,
        email=student.email,
        phone=student.phone,
        school_id=school_seq(db, student.school_id),
        grade=grade,
        section=section,
        curriculum=curriculum,
        learning_type=learning_type,
        learning_teacher=learning_teacher,
        learning_teacher_ids=learning_teacher_ids,
        teacher_guided_subjects=guided,
        self_learned_subjects=self_learned,
        password_status=_password_status(student, has_logged_in=has_logged_in),
        is_active=student.is_active,
        created_at=student.created_at,
        updated_at=student.updated_at,
    )


def _scope_school(actor: User, school_id: int | None) -> int | None:
    if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR):
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        return actor.school_id
    if actor.role == Role.MASTER_ADMIN:
        return school_id
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def _base_student_query(school_id: int | None):
    stmt = select(User).where(User.role == Role.STUDENT)
    if school_id is not None:
        stmt = stmt.where(User.school_id == school_id)
    return stmt


def student_options(db: Session, actor: User) -> StudentOptionsResponse:
    if actor.role == Role.TUTOR:
        assigned = list_tutor_assigned_students(db, actor)
        grades: list[str] = []
        sections: list[str] = []
        curricula: list[str] = []
        seen_g: set[str] = set()
        seen_s: set[str] = set()
        seen_c: set[str] = set()
        for student in assigned:
            grade, section, curriculum = _class_fields(student)
            if grade and grade not in seen_g:
                seen_g.add(grade)
                grades.append(grade)
            if section and section not in seen_s:
                seen_s.add(section)
                sections.append(section)
            if curriculum and curriculum.lower() not in seen_c:
                seen_c.add(curriculum.lower())
                curricula.append(curriculum)
        grades.sort(key=lambda g: (len(g), g))
        sections.sort()
        curricula.sort(key=str.lower)
        return StudentOptionsResponse(
            grades=grades,
            sections=sections,
            curricula=curricula,
            learning_types=list(LEARNING_TYPES),
            default_limit=DEFAULT_PAGE_LIMIT,
            page_size_options=list(PAGE_SIZE_OPTIONS),
        )

    school_id = (
        _scope_school(actor, None)
        if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR)
        else actor.school_id
    )
    grades = []
    sections = []
    curricula = []
    if school_id:
        classes = list(
            db.scalars(
                select(SchoolClass)
                .where(SchoolClass.school_id == school_id)
                .order_by(SchoolClass.grade, SchoolClass.section)
            )
        )
        seen_g = set()
        seen_s = set()
        seen_c = set()
        for row in classes:
            g = str(row.grade).strip()
            s = str(row.section).strip().upper()
            c = str(row.curriculum).strip()
            if g and g not in seen_g:
                seen_g.add(g)
                grades.append(g)
            if s and s not in seen_s:
                seen_s.add(s)
                sections.append(s)
            if c and c.lower() not in seen_c:
                seen_c.add(c.lower())
                curricula.append(c)
    return StudentOptionsResponse(
        grades=grades,
        sections=sections,
        curricula=curricula,
        learning_types=list(LEARNING_TYPES),
        default_limit=DEFAULT_PAGE_LIMIT,
        page_size_options=list(PAGE_SIZE_OPTIONS),
    )


def _matches_student_filters(
    student: User,
    *,
    q: str | None,
    grade: str | None,
    section: str | None,
    curriculum: str | None,
    status_filter: str | None,
) -> bool:
    if status_filter == "active" and not student.is_active:
        return False
    if status_filter == "inactive" and student.is_active:
        return False
    if q:
        term = q.strip().lower()
        hay = f"{student.full_name} {student.email} {student.phone or ''}".lower()
        if term not in hay:
            return False
    s_grade, s_section, s_curriculum = _class_fields(student)
    if grade and grade.lower() != "all" and (s_grade or "") != grade.strip():
        return False
    if section and section.lower() != "all" and (s_section or "").upper() != section.strip().upper():
        return False
    if curriculum and curriculum.lower() != "all":
        cur = curriculum.strip().lower()
        board = (student.teaching_board or "").strip().lower()
        if (s_curriculum or "").lower() != cur and board != cur:
            return False
    return True


def _filtered_student_page(
    db: Session,
    assigned: list[User],
    *,
    q: str | None = None,
    grade: str | None = None,
    section: str | None = None,
    curriculum: str | None = None,
    learning_type: str | None = None,
    status_filter: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> StudentListResponse:
    filtered = [
        s
        for s in assigned
        if _matches_student_filters(
            s,
            q=q,
            grade=grade,
            section=section,
            curriculum=curriculum,
            status_filter=status_filter,
        )
    ]
    logged_in = _logged_in_user_ids(db, [s.id for s in filtered])
    items = [_student_response(db, s, has_logged_in=s.id in logged_in) for s in filtered]
    if learning_type and learning_type.lower() != "all":
        want = learning_type.strip().lower()
        items = [i for i in items if i.learning_type.lower() == want]
    total = len(items)
    page = items[offset : offset + limit]
    return StudentListResponse(
        items=page,
        meta=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            default_limit=DEFAULT_PAGE_LIMIT,
        ),
    )


def _list_tutor_scoped_students(
    db: Session,
    actor: User,
    *,
    q: str | None = None,
    grade: str | None = None,
    section: str | None = None,
    curriculum: str | None = None,
    learning_type: str | None = None,
    status_filter: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> StudentListResponse:
    """School tutors only see students tagged to them (class match minus exclusions)."""
    assigned = list_tutor_assigned_students(db, actor)
    return _filtered_student_page(
        db,
        assigned,
        q=q,
        grade=grade,
        section=section,
        curriculum=curriculum,
        learning_type=learning_type,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )


def _list_individual_tutor_students(
    db: Session,
    actor: User,
    *,
    q: str | None = None,
    grade: str | None = None,
    section: str | None = None,
    curriculum: str | None = None,
    learning_type: str | None = None,
    status_filter: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> StudentListResponse:
    """Individual tutors manage students they created (no school)."""
    assigned = list(
        db.scalars(
            select(User)
            .where(
                User.role == Role.STUDENT,
                User.created_by == actor.id,
                User.school_id.is_(None),
            )
            .order_by(User.created_at.desc())
        )
    )
    return _filtered_student_page(
        db,
        assigned,
        q=q,
        grade=grade,
        section=section,
        curriculum=curriculum,
        learning_type=learning_type,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )


def _assert_tutor_assigned_student(db: Session, actor: User, student: User) -> None:
    if actor.role != Role.TUTOR:
        return
    assigned_ids = {s.id for s in list_tutor_assigned_students(db, actor)}
    if student.id not in assigned_ids:
        raise AuthException("Student is not assigned to you.", status.HTTP_403_FORBIDDEN)


def list_students(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
    q: str | None = None,
    grade: str | None = None,
    section: str | None = None,
    curriculum: str | None = None,
    learning_type: str | None = None,
    status_filter: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> StudentListResponse:
    limit = min(max(1, limit), MAX_PAGE_LIMIT)
    offset = max(0, offset)
    if actor.role == Role.TUTOR:
        list_fn = (
            _list_individual_tutor_students
            if is_individual_tutor(actor)
            else _list_tutor_scoped_students
        )
        return list_fn(
            db,
            actor,
            q=q,
            grade=grade,
            section=section,
            curriculum=curriculum,
            learning_type=learning_type,
            status_filter=status_filter,
            limit=limit,
            offset=offset,
        )
    scoped = _scope_school(actor, school_id)
    stmt = _base_student_query(scoped)

    if q:
        term = f"%{q.strip().lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(User.full_name).like(term),
                func.lower(User.email).like(term),
                func.lower(User.phone).like(term),
            )
        )

    # ponytail: JSON text match; upgrade to jsonb path ops when class shape is normalized.
    classes_text = func.lower(cast(User.teaching_classes, String))
    if grade and grade.lower() != "all":
        g = grade.strip().lower()
        stmt = stmt.where(classes_text.like(f'%"grade": "{g}"%'))
    if section and section.lower() != "all":
        sec = section.strip().upper()
        stmt = stmt.where(
            or_(
                classes_text.like(f'%"{sec.lower()}"%'),
                cast(User.teaching_classes, String).like(f'%"{sec}"%'),
            )
        )
    if curriculum and curriculum.lower() != "all":
        cur = curriculum.strip().lower()
        stmt = stmt.where(
            or_(
                func.lower(User.teaching_board) == cur,
                classes_text.like(f'%"curriculum": "{cur}"%'),
            )
        )

    if status_filter == "active":
        stmt = stmt.where(User.is_active.is_(True))
    elif status_filter == "inactive":
        stmt = stmt.where(User.is_active.is_(False))

    need_learning_filter = bool(learning_type and learning_type.lower() != "all")

    if need_learning_filter:
        # ponytail: learning_type is derived from tutor overlap; filter in memory then page.
        students = list(db.scalars(stmt.order_by(User.created_at.desc())))
        logged_in = _logged_in_user_ids(db, [s.id for s in students])
        items = [
            _student_response(db, s, has_logged_in=s.id in logged_in) for s in students
        ]
        want = learning_type.strip().lower()
        items = [i for i in items if i.learning_type.lower() == want]
        total = len(items)
        page = items[offset : offset + limit]
        return StudentListResponse(
            items=page,
            meta=PaginationMeta(
                total=total,
                limit=limit,
                offset=offset,
                default_limit=DEFAULT_PAGE_LIMIT,
            ),
        )

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = int(db.scalar(count_stmt) or 0)
    students = list(db.scalars(stmt.order_by(User.created_at.desc()).offset(offset).limit(limit)))
    logged_in = _logged_in_user_ids(db, [s.id for s in students])
    return StudentListResponse(
        items=[_student_response(db, s, has_logged_in=s.id in logged_in) for s in students],
        meta=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            default_limit=DEFAULT_PAGE_LIMIT,
        ),
    )


def get_student(db: Session, actor: User, student_id: int) -> StudentResponse:
    student = require_user_by_account_id(db, student_id)
    assert_actor_manages_student(actor, student)
    return _student_response(db, student)


def _set_student_class(student: User, school_class: SchoolClass) -> None:
    student.teaching_board = school_class.curriculum
    student.teaching_classes = [
        {
            "grade": school_class.grade,
            "sections": [school_class.section.strip().upper()],
            "curriculum": school_class.curriculum,
            "school_class_id": school_class.seq,
        }
    ]


def _attach_creating_tutor_class(student: User, actor: User) -> None:
    """When a school tutor creates a student, stamp the tutor's first class so roster/session matching works.
    ponytail: first teaching_classes entry only if tutor has several — upgrade: pick class in add dialog.
    """
    if actor.role != Role.TUTOR or not actor.school_id:
        return
    raw = actor.teaching_classes
    if not isinstance(raw, list) or not raw:
        return
    first = raw[0]
    if not isinstance(first, dict):
        return
    grade = str(first.get("grade", "")).strip()
    sections = [str(s).strip().upper() for s in (first.get("sections") or []) if str(s).strip()]
    if not grade or not sections:
        return
    entry: dict = {"grade": grade, "sections": [sections[0]]}
    if first.get("curriculum"):
        entry["curriculum"] = first["curriculum"]
    sc_id = first.get("school_class_id")
    if sc_id is not None and str(sc_id).strip().lstrip("-").isdigit():
        entry["school_class_id"] = int(sc_id)
    student.teaching_board = entry.get("curriculum") or None
    student.teaching_classes = [entry]


def _create_student_row(
    db: Session,
    actor: User,
    item: StudentCreateItem,
    school_id: int | None,
) -> User:
    email = str(item.parent_email).lower()

    roll = normalize_username(item.roll_number) or item.roll_number.strip().lower()
    if username_taken(db, roll):
        raise AuthException("Roll number (user id) already in use.", status.HTTP_409_CONFLICT)
    ensure_username_available(db, roll)

    password = secrets.token_urlsafe(12)
    user = User(
        full_name=item.student_name.strip(),
        email=email,
        phone=normalize_phone(item.parent_phone),
        password_hash=hash_password(password),
        role=Role.STUDENT,
        school_id=school_id,
        is_active=True,
        is_verified=False,
        created_by=actor.id,
    )
    _attach_creating_tutor_class(user, actor)
    db.add(user)
    db.flush()

    settings = get_or_create_user_settings(db, user)
    settings.username = roll
    db.flush()
    return user


def create_student(
    db: Session,
    actor: User,
    item: StudentCreateItem,
    *,
    school_id: int | None = None,
) -> StudentResponse:
    target = None if is_individual_tutor(actor) else resolve_target_school_id(actor, school_id=school_id)
    student = _create_student_row(db, actor, item, target)
    return _student_response(db, student)


def bulk_create_students(
    db: Session,
    actor: User,
    items: list[StudentCreateItem],
    *,
    school_id: int | None = None,
) -> StudentBulkCreateResponse:
    if len(items) > MAX_BULK_IMPORT:
        raise AuthException(
            f"At most {MAX_BULK_IMPORT} students per import.",
            status.HTTP_400_BAD_REQUEST,
        )
    target = None if is_individual_tutor(actor) else resolve_target_school_id(actor, school_id=school_id)
    created: list[StudentResponse] = []
    errors: list[BulkRowError] = []
    seen_rolls: set[str] = set()

    for index, item in enumerate(items, start=1):
        email_key = str(item.parent_email).lower()
        roll_key = (normalize_username(item.roll_number) or item.roll_number.strip().lower())
        if roll_key in seen_rolls:
            errors.append(
                BulkRowError(row=index, email=email_key, roll_number=roll_key, message="Duplicate roll number in file.")
            )
            continue
        seen_rolls.add(roll_key)
        try:
            student = _create_student_row(db, actor, item, target)
            created.append(_student_response(db, student))
        except AuthException as exc:
            errors.append(BulkRowError(row=index, email=email_key, roll_number=roll_key, message=exc.detail))
        except Exception as exc:  # ponytail: surface row errors without aborting whole import
            errors.append(BulkRowError(row=index, email=email_key, roll_number=roll_key, message=str(exc)))

    return StudentBulkCreateResponse(created=created, errors=errors)


def update_student(
    db: Session,
    actor: User,
    student_id: int,
    payload: StudentUpdateRequest,
) -> StudentResponse:
    student = require_user_by_account_id(db, student_id)
    assert_actor_manages_student(actor, student)
    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise AuthException("No fields to update.", status.HTTP_400_BAD_REQUEST)

    if "parent_email" in data and data["parent_email"] is not None:
        student.email = str(data["parent_email"]).lower()

    if "student_name" in data and data["student_name"] is not None:
        student.full_name = data["student_name"].strip()

    if "parent_phone" in data:
        if data["parent_phone"]:
            student.phone = normalize_phone(data["parent_phone"])
        else:
            student.phone = None

    if "is_active" in data and data["is_active"] is not None:
        student.is_active = bool(data["is_active"])

    if "class_id" in data and data["class_id"] is not None:
        if not student.school_id:
            raise AuthException("Student has no school.", status.HTTP_400_BAD_REQUEST)
        school_class = db.scalar(
            select(SchoolClass).where(
                SchoolClass.school_id == student.school_id,
                SchoolClass.seq == int(data["class_id"]),
            )
        )
        if not school_class:
            raise AuthException("Class not found.", status.HTTP_404_NOT_FOUND)
        _set_student_class(student, school_class)
        from app.modules.student_learning.service import reset_student_learning

        reset_student_learning(db, student.id)
    elif any(k in data for k in ("grade", "section", "curriculum")):
        grade = data.get("grade") or (_class_fields(student)[0] or "")
        section = (data.get("section") or (_class_fields(student)[1] or "")).upper()
        curriculum = data.get("curriculum") or (_class_fields(student)[2] or student.teaching_board or "")
        if not grade or not section:
            raise AuthException("grade and section are required.", status.HTTP_400_BAD_REQUEST)
        student.teaching_board = curriculum or None
        student.teaching_classes = [
            {
                "grade": grade,
                "sections": [section],
                "curriculum": curriculum or None,
            }
        ]
        from app.modules.student_learning.service import reset_student_learning

        reset_student_learning(db, student.id)

    student.updated_at = datetime.now(UTC)
    db.flush()
    return _student_response(db, student)


def delete_student(db: Session, actor: User, student_id: int) -> None:
    student = require_user_by_account_id(db, student_id)
    assert_actor_manages_student(actor, student)
    db.delete(student)
    db.flush()


def move_students_to_class(
    db: Session,
    actor: User,
    payload: StudentMoveClassRequest,
) -> StudentActionResponse:
    if is_individual_tutor(actor):
        school_class = db.scalar(
            select(SchoolClass).where(
                SchoolClass.owner_user_id == actor.id,
                SchoolClass.school_id.is_(None),
                SchoolClass.seq == payload.class_id,
            )
        )
    else:
        school_id = resolve_target_school_id(actor)
        school_class = db.scalar(
            select(SchoolClass).where(
                SchoolClass.school_id == school_id,
                SchoolClass.seq == payload.class_id,
            )
        )
    if not school_class:
        raise AuthException("Class not found.", status.HTTP_404_NOT_FOUND)

    from app.modules.student_learning.service import reset_student_learning

    updated = 0
    for sid in payload.student_ids:
        student = require_user_by_account_id(db, sid)
        assert_actor_manages_student(actor, student)
        _assert_tutor_assigned_student(db, actor, student)
        _set_student_class(student, school_class)
        student.updated_at = datetime.now(UTC)
        # Class change → restart learning stats for this student
        reset_student_learning(db, student.id)
        updated += 1
    db.flush()
    return StudentActionResponse(
        message=f"Moved {updated} student(s) to {school_class.grade}-{school_class.section}.",
        updated=updated,
    )


def _assign_one_teacher_to_students(
    db: Session,
    actor: User,
    *,
    school_id,
    teacher: User,
    student_ids: list[int],
) -> int:
    if teacher.role != Role.TUTOR or teacher.school_id != school_id:
        raise AuthException("Teacher not found in this school.", status.HTTP_404_NOT_FOUND)
    if not teacher.is_active:
        raise AuthException(f"Teacher {teacher.full_name} is inactive.", status.HTTP_400_BAD_REQUEST)

    classes = list(teacher.teaching_classes) if isinstance(teacher.teaching_classes, list) else []
    seen = {
        (
            str(c.get("grade", "")).strip(),
            str((c.get("sections") or [" "])[0]).strip().upper() if isinstance(c, dict) else "",
            str(c.get("curriculum", "")).strip().lower() if isinstance(c, dict) else "",
        )
        for c in classes
        if isinstance(c, dict)
    }

    excluded = {
        int(x)
        for x in (teacher.excluded_student_ids or [])
        if str(x).strip().lstrip("-").isdigit()
    }

    updated = 0
    for sid in student_ids:
        student = require_user_by_account_id(db, sid)
        assert_actor_manages_student(actor, student)
        _assert_tutor_assigned_student(db, actor, student)
        grade, section, curriculum = _class_fields(student)
        if not grade or not section:
            raise AuthException(
                f"Student {student.full_name} has no class. Move them into a class first.",
                status.HTTP_400_BAD_REQUEST,
            )
        key = (grade, section, (curriculum or "").lower())
        if key not in seen:
            entry = {
                "grade": grade,
                "sections": [section],
                "curriculum": curriculum,
            }
            raw = student.teaching_classes
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) and raw[0].get("school_class_id") is not None:
                entry["school_class_id"] = raw[0]["school_class_id"]
            classes.append(entry)
            seen.add(key)
        excluded.discard(student.id)
        updated += 1

    teacher.teaching_classes = classes or None
    teacher.excluded_student_ids = sorted(excluded) or None
    teacher.updated_at = datetime.now(UTC)
    return updated


def assign_teacher_to_students(
    db: Session,
    actor: User,
    payload: StudentAssignTeacherRequest,
) -> StudentActionResponse:
    if is_individual_tutor(actor):
        raise AuthException("Not available for individual teachers.", status.HTTP_403_FORBIDDEN)
    school_id = resolve_target_school_id(actor)
    teacher_names: list[str] = []
    link_count = 0
    notified: list[int] = []
    for tid in payload.teacher_ids:
        teacher = require_user_by_account_id(db, tid)
        link_count += _assign_one_teacher_to_students(
            db,
            actor,
            school_id=school_id,
            teacher=teacher,
            student_ids=payload.student_ids,
        )
        teacher_names.append(teacher.full_name)
        notified.append(teacher.id)
    db.flush()
    if notified:
        n = len(payload.student_ids)
        notify_users(
            db,
            recipient_ids=notified,
            type="student_tagged",
            title="Students tagged to you",
            body=f"{n} student(s) were tagged to you.",
            actor_id=actor.id,
            school_id=school_id,
            link="/tutor/students",
        )
    names = ", ".join(teacher_names)
    return StudentActionResponse(
        message=f"Assigned {names} to {len(payload.student_ids)} student(s).",
        updated=link_count,
    )


def export_students_csv(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
    q: str | None = None,
    grade: str | None = None,
    section: str | None = None,
    curriculum: str | None = None,
    learning_type: str | None = None,
    status_filter: str | None = None,
) -> str:
    result = list_students(
        db,
        actor,
        school_id=school_id,
        q=q,
        grade=grade,
        section=section,
        curriculum=curriculum,
        learning_type=learning_type,
        status_filter=status_filter,
        limit=MAX_PAGE_LIMIT,
        offset=0,
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "user_id",
            "full_name",
            "email",
            "phone",
            "grade",
            "section",
            "curriculum",
            "learning_type",
            "learning_teacher",
            "password_status",
            "status",
        ]
    )
    for s in result.items:
        writer.writerow(
            [
                s.user_id,
                s.full_name,
                s.email,
                s.phone or "",
                s.grade or "",
                s.section or "",
                s.curriculum or "",
                s.learning_type,
                s.learning_teacher or "",
                s.password_status,
                "Active" if s.is_active else "Inactive",
            ]
        )
    return buf.getvalue()

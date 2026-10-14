from __future__ import annotations

import csv
import io
import secrets
from datetime import UTC, datetime

from fastapi import status
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.public_ids import require_user_by_account_id, school_seq
from app.modules.auth.service import get_or_create_user_settings, list_tutor_assigned_students
from app.modules.auth.signup.enums import TeachingSubjectEnum
from app.modules.auth.signup.validators import (
    normalize_phone,
)
from app.modules.school_admin.classes.models import SchoolClass, SchoolSubject
from app.modules.notifications.service import notify_users
from app.modules.student_learning.enrollment import enrolled_subject_names
from app.modules.school_admin.teachers.constants import DEFAULT_PAGE_LIMIT, MAX_BULK_IMPORT, MAX_PAGE_LIMIT, PAGE_SIZE_OPTIONS
from app.modules.school_admin.teachers.schemas import (
    BulkRowError,
    PaginationMeta,
    TeacherAssignResponse,
    TeacherBulkCreateResponse,
    TeacherClassBrief,
    TeacherCreateItem,
    TeacherDetailResponse,
    TeacherGradeSubjects,
    TeacherListResponse,
    TeacherOptionsResponse,
    TeacherResponse,
    TeacherStudentBrief,
    TeacherSubjectBrief,
    TeacherUnassignRequest,
    TeacherUpdateRequest,
)
from app.modules.school_admin.teachers.validators import (
    assert_actor_manages_teacher,
    generate_unique_username,
    resolve_target_school_id,
    username_taken,
)
from app.modules.users.models import User, UserSettings


def _grade_section_labels(item: dict) -> list[str]:
    """Build display labels for one teaching_classes entry (one per section)."""
    grade = str(item.get("grade", "")).strip()
    sections = item.get("sections") or []
    curriculum = str(item.get("curriculum", "")).strip()
    if not grade:
        return []
    if not sections:
        return [f"{grade} · {curriculum}" if curriculum else grade]
    labels: list[str] = []
    for section in sections:
        sec = str(section).strip().upper()
        if not sec:
            continue
        if curriculum:
            labels.append(f"{grade}-{sec} · {curriculum}")
        else:
            labels.append(f"{grade}-{sec}")
    return labels


def _grades_label(tutor: User) -> str | None:
    raw = tutor.teaching_classes
    if not isinstance(raw, list) or not raw:
        return None
    labels: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            for label in _grade_section_labels(item):
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                labels.append(label)
        else:
            g = str(item).strip()
            if g and g.lower() not in seen:
                seen.add(g.lower())
                labels.append(g)
    return ", ".join(labels) if labels else None


def _resolve_school_class(
    db: Session,
    tutor: User,
    item: dict,
    *,
    section: str | None = None,
) -> SchoolClass | None:
    grade = str(item.get("grade", "")).strip()
    sections = item.get("sections") or []
    sec = (section or (str(sections[0]).strip().upper() if sections else "")).strip().upper()
    curriculum = str(item.get("curriculum", "")).strip()
    if tutor.school_id and grade and sec:
        stmt = select(SchoolClass).where(
            SchoolClass.school_id == tutor.school_id,
            SchoolClass.grade == grade,
            func.upper(SchoolClass.section) == sec,
        )
        if curriculum:
            stmt = stmt.where(func.lower(SchoolClass.curriculum) == curriculum.lower())
        match = db.scalar(stmt)
        if match:
            return match
    class_id = item.get("school_class_id")
    if str(class_id).isdigit() and tutor.school_id:
        return db.scalar(
            select(SchoolClass).where(
                SchoolClass.school_id == tutor.school_id,
                SchoolClass.seq == int(class_id),
            )
        )
    return None


def _subjects_for_class(
    db: Session,
    tutor: User,
    item: dict,
    teacher_subjects: list[str],
    *,
    section: str | None = None,
) -> str:
    """Teacher subjects offered in this class (∩ class mapping); else all teacher subjects."""
    if not teacher_subjects:
        return "—"
    school_class = _resolve_school_class(db, tutor, item, section=section)
    if school_class is None:
        return ", ".join(teacher_subjects)
    mapped = enrolled_subject_names(db, school_class)
    if not mapped:
        return ", ".join(teacher_subjects)
    mapped_keys = {n.strip().lower() for n in mapped}
    matched = [n for n in teacher_subjects if n.strip().lower() in mapped_keys]
    return ", ".join(matched) if matched else "—"


def _assignment_rows(
    db: Session,
    tutor: User,
    *,
    teacher_subjects: list[str] | None = None,
) -> list[TeacherGradeSubjects]:
    """One row per grade/section with subjects for nested teacher table view."""
    if teacher_subjects is None:
        teacher_subjects = [s.name for s in _teacher_subjects(db, tutor)]
    raw = tutor.teaching_classes
    if not isinstance(raw, list) or not raw:
        if teacher_subjects:
            return [TeacherGradeSubjects(grade="—", subjects=", ".join(teacher_subjects))]
        return []

    rows: list[TeacherGradeSubjects] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            g = str(item).strip()
            if not g or g.lower() in seen:
                continue
            seen.add(g.lower())
            rows.append(
                TeacherGradeSubjects(
                    grade=g,
                    subjects=", ".join(teacher_subjects) if teacher_subjects else "—",
                )
            )
            continue

        grade = str(item.get("grade", "")).strip()
        sections = item.get("sections") or []
        curriculum = str(item.get("curriculum", "")).strip()
        if not grade:
            continue

        if not sections:
            label = f"{grade} · {curriculum}" if curriculum else grade
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                TeacherGradeSubjects(
                    grade=label,
                    subjects=_subjects_for_class(db, tutor, item, teacher_subjects),
                )
            )
            continue

        for section in sections:
            sec = str(section).strip().upper()
            if not sec:
                continue
            label = f"{grade}-{sec} · {curriculum}" if curriculum else f"{grade}-{sec}"
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                TeacherGradeSubjects(
                    grade=label,
                    subjects=_subjects_for_class(
                        db, tutor, item, teacher_subjects, section=sec
                    ),
                )
            )
    return rows


def _username_for_user(db: Session, user_id: int) -> str:
    row = db.get(UserSettings, user_id)
    if row and row.username:
        return row.username
    user = db.get(User, user_id)
    if user:
        settings = get_or_create_user_settings(db, user)
        return settings.username or user.email.split("@", 1)[0]
    return ""


def _teacher_response(db: Session, tutor: User, *, assigned_students: int | None = None) -> TeacherResponse:
    if assigned_students is None:
        assigned_students = len(list_tutor_assigned_students(db, tutor)) if tutor.school_id else 0
    live = _teacher_subjects(db, tutor)
    subject_names = [s.name for s in live]
    subject = ", ".join(subject_names) if subject_names else None
    return TeacherResponse(
        id=tutor.id,
        user_id=_username_for_user(db, tutor.id),
        full_name=tutor.full_name,
        email=tutor.email,
        phone=tutor.phone,
        school_id=school_seq(db, tutor.school_id),
        subject=subject,
        grades=_grades_label(tutor),
        assignments=_assignment_rows(db, tutor, teacher_subjects=subject_names),
        assigned_students=assigned_students,
        is_active=tutor.is_active,
        last_login=tutor.updated_at,
        created_at=tutor.created_at,
        updated_at=tutor.updated_at,
    )


def _base_tutor_query(school_id: int | None):
    stmt = select(User).where(User.role == Role.TUTOR)
    if school_id is not None:
        stmt = stmt.where(User.school_id == school_id)
    return stmt


def list_teachers(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
    q: str | None = None,
    subject: str | None = None,
    curriculum: str | None = None,
    status_filter: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> TeacherListResponse:
    limit = min(max(1, limit), MAX_PAGE_LIMIT)
    offset = max(0, offset)

    scoped_school: int | None
    if actor.role in (Role.SCHOOL_ADMIN, Role.TUTOR):
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        scoped_school = actor.school_id
    elif actor.role == Role.MASTER_ADMIN:
        scoped_school = school_id
    else:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    stmt = _base_tutor_query(scoped_school)

    if q:
        term = f"%{q.strip().lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(User.full_name).like(term),
                func.lower(User.email).like(term),
                func.lower(User.phone).like(term),
            )
        )

    if subject and subject.lower() != "all":
        subjects_json = cast(User.teaching_subjects, JSONB)
        stmt = stmt.where(subjects_json.contains([subject.strip()]))

    if curriculum and curriculum.lower() != "all":
        cur = curriculum.strip().lower()
        classes_text = func.lower(cast(User.teaching_classes, String))
        stmt = stmt.where(classes_text.like(f'%"curriculum": "{cur}"%'))

    if status_filter == "active":
        stmt = stmt.where(User.is_active.is_(True))
    elif status_filter == "inactive":
        stmt = stmt.where(User.is_active.is_(False))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = int(db.scalar(count_stmt) or 0)

    tutors = list(
        db.scalars(stmt.order_by(User.created_at.desc()).offset(offset).limit(limit))
    )

    # Batch assigned-student counts for the page (avoid N full school scans looking empty).
    items = [_teacher_response(db, t) for t in tutors]
    return TeacherListResponse(
        items=items,
        meta=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            default_limit=DEFAULT_PAGE_LIMIT,
        ),
    )


def _teacher_subjects(db: Session, tutor: User) -> list[TeacherSubjectBrief]:
    if not tutor.school_id:
        return []
    names = {
        str(n).strip().lower()
        for n in (tutor.teaching_subjects or [])
        if str(n).strip()
    }
    if not names:
        return []
    subjects = list(
        db.scalars(select(SchoolSubject).where(SchoolSubject.school_id == tutor.school_id).order_by(SchoolSubject.seq))
    )
    return [
        TeacherSubjectBrief(id=s.seq, name=s.name, code=s.code)
        for s in subjects
        if s.name.strip().lower() in names
    ]


def _teacher_classes(db: Session, tutor: User) -> list[TeacherClassBrief]:
    raw = tutor.teaching_classes
    if not isinstance(raw, list) or not raw:
        return []
    out: list[TeacherClassBrief] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        class_id = item.get("school_class_id")
        grade = str(item.get("grade", "")).strip()
        sections = item.get("sections") or []
        curriculum = str(item.get("curriculum", "")).strip()
        section = str(sections[0]).strip().upper() if sections else ""
        seq = int(class_id) if str(class_id).isdigit() else None
        if seq is None and tutor.school_id and grade and section:
            stmt = select(SchoolClass).where(
                SchoolClass.school_id == tutor.school_id,
                SchoolClass.grade == grade,
                func.upper(SchoolClass.section) == section,
            )
            if curriculum:
                stmt = stmt.where(func.lower(SchoolClass.curriculum) == curriculum.lower())
            match = db.scalar(stmt)
            if match:
                seq = match.seq
                curriculum = match.curriculum
        if seq is None:
            continue
        key = str(seq)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            TeacherClassBrief(
                id=seq,
                grade=grade or "—",
                section=section or "—",
                curriculum=curriculum or "—",
            )
        )
    return out


def _teacher_students(db: Session, tutor: User) -> list[TeacherStudentBrief]:
    students = list_tutor_assigned_students(db, tutor) if tutor.school_id else []
    out: list[TeacherStudentBrief] = []
    for student in students:
        grade = None
        section = None
        raw = student.teaching_classes
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            grade = str(raw[0].get("grade", "")).strip() or None
            secs = raw[0].get("sections") or []
            section = str(secs[0]).strip().upper() if secs else None
        out.append(
            TeacherStudentBrief(
                id=student.id,
                full_name=student.full_name,
                email=student.email,
                grade=grade,
                section=section,
            )
        )
    return out


def _teacher_detail(db: Session, tutor: User) -> TeacherDetailResponse:
    base = _teacher_response(db, tutor)
    return TeacherDetailResponse(
        **base.model_dump(),
        subjects=_teacher_subjects(db, tutor),
        classes=_teacher_classes(db, tutor),
        students=_teacher_students(db, tutor),
    )


def get_teacher(db: Session, actor: User, teacher_id: int) -> TeacherDetailResponse:
    tutor = require_user_by_account_id(db, teacher_id)
    assert_actor_manages_teacher(actor, tutor)
    return _teacher_detail(db, tutor)


def unassign_teacher_items(
    db: Session,
    actor: User,
    teacher_id: int,
    payload: TeacherUnassignRequest,
) -> TeacherDetailResponse:
    tutor = require_user_by_account_id(db, teacher_id)
    assert_actor_manages_teacher(actor, tutor)

    if payload.subject_ids:
        remove_names: set[str] = set()
        if tutor.school_id:
            for subject in db.scalars(
                select(SchoolSubject).where(
                    SchoolSubject.school_id == tutor.school_id,
                    SchoolSubject.seq.in_(payload.subject_ids),
                )
            ):
                remove_names.add(subject.name.strip().lower())
        kept = [
            n
            for n in (tutor.teaching_subjects or [])
            if str(n).strip().lower() not in remove_names
        ]
        tutor.teaching_subjects = kept or None

    if payload.class_ids:
        remove_ids = {str(i) for i in payload.class_ids}
        kept_classes: list[dict] = []
        for item in tutor.teaching_classes or []:
            if not isinstance(item, dict):
                continue
            cid = item.get("school_class_id")
            if str(cid) in remove_ids:
                continue
            kept_classes.append(dict(item))
        tutor.teaching_classes = kept_classes or None

    if payload.student_ids:
        existing = {
            int(x)
            for x in (tutor.excluded_student_ids or [])
            if str(x).strip().lstrip("-").isdigit()
        }
        existing.update(int(i) for i in payload.student_ids)
        tutor.excluded_student_ids = sorted(existing)

    tutor.updated_at = datetime.now(UTC)
    db.flush()
    return _teacher_detail(db, tutor)


def _create_teacher_row(
    db: Session,
    actor: User,
    item: TeacherCreateItem,
    school_id: int,
) -> User:
    username = generate_unique_username(db, item.full_name, str(item.email))
    password = secrets.token_urlsafe(12)

    user = User(
        full_name=item.full_name.strip(),
        email=str(item.email).lower(),
        phone=normalize_phone(item.phone),
        password_hash=hash_password(password),
        role=Role.TUTOR,
        school_id=school_id,
        is_active=True,
        is_verified=False,
        created_by=actor.id,
    )
    db.add(user)
    db.flush()

    settings = get_or_create_user_settings(db, user)
    settings.username = username
    db.flush()
    return user


def create_teacher(
    db: Session,
    actor: User,
    item: TeacherCreateItem,
    *,
    school_id: int | None = None,
) -> TeacherResponse:
    target_school = resolve_target_school_id(actor, school_id=school_id)
    tutor = _create_teacher_row(db, actor, item, target_school)
    notify_users(
        db,
        recipient_ids=[tutor.id],
        type="teacher_created",
        title="Welcome to the school",
        body="You were added as a teacher. Sign in to view your classes and students.",
        actor_id=actor.id,
        school_id=target_school,
        link="/tutor/dashboard",
    )
    return _teacher_response(db, tutor)


def bulk_create_teachers(
    db: Session,
    actor: User,
    items: list[TeacherCreateItem],
    *,
    school_id: int | None = None,
) -> TeacherBulkCreateResponse:
    if len(items) > MAX_BULK_IMPORT:
        raise AuthException(
            f"At most {MAX_BULK_IMPORT} teachers per import.",
            status.HTTP_400_BAD_REQUEST,
        )
    target_school = resolve_target_school_id(actor, school_id=school_id)

    created: list[TeacherResponse] = []
    errors: list[BulkRowError] = []
    new_ids: list[int] = []

    for index, item in enumerate(items, start=1):
        email_key = str(item.email).lower()

        try:
            tutor = _create_teacher_row(db, actor, item, target_school)
            created.append(_teacher_response(db, tutor))
            new_ids.append(tutor.id)
        except AuthException as exc:
            errors.append(BulkRowError(row=index, email=email_key, message=exc.detail))
        except Exception as exc:  # ponytail: surface row errors without aborting whole import
            errors.append(BulkRowError(row=index, email=email_key, message=str(exc)))

    if new_ids:
        notify_users(
            db,
            recipient_ids=new_ids,
            type="teacher_created",
            title="Welcome to the school",
            body="You were added as a teacher. Sign in to view your classes and students.",
            actor_id=actor.id,
            school_id=target_school,
            link="/tutor/dashboard",
        )

    return TeacherBulkCreateResponse(created=created, errors=errors)


def _park_teacher_assignments(tutor: User) -> None:
    """Save assigns then clear them so inactive teachers show no subjects/classes/students."""
    classes = tutor.teaching_classes if isinstance(tutor.teaching_classes, list) else []
    tutor.assignment_snapshot = {
        "teaching_subjects": list(tutor.teaching_subjects or []),
        "teaching_board": tutor.teaching_board,
        "teaching_classes": [dict(c) if isinstance(c, dict) else c for c in classes],
        "excluded_student_ids": list(tutor.excluded_student_ids or []),
    }
    tutor.teaching_subjects = None
    tutor.teaching_board = None
    tutor.teaching_classes = None
    tutor.excluded_student_ids = None


def _restore_teacher_assignments(tutor: User) -> None:
    """Bring back parked assigns when the teacher is activated again."""
    snap = tutor.assignment_snapshot
    if not isinstance(snap, dict):
        return
    subjects = snap.get("teaching_subjects")
    tutor.teaching_subjects = list(subjects) if isinstance(subjects, list) and subjects else None
    board = snap.get("teaching_board")
    tutor.teaching_board = str(board).strip() if board else None
    classes = snap.get("teaching_classes")
    tutor.teaching_classes = list(classes) if isinstance(classes, list) and classes else None
    excluded = snap.get("excluded_student_ids")
    tutor.excluded_student_ids = (
        [int(x) for x in excluded if str(x).strip().lstrip("-").isdigit()]
        if isinstance(excluded, list) and excluded
        else None
    )
    tutor.assignment_snapshot = None


def update_teacher(
    db: Session,
    actor: User,
    teacher_id: int,
    payload: TeacherUpdateRequest,
) -> TeacherDetailResponse:
    tutor = require_user_by_account_id(db, teacher_id)
    assert_actor_manages_teacher(actor, tutor)

    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise AuthException("No fields to update.", status.HTTP_400_BAD_REQUEST)

    if "email" in data and data["email"] is not None:
        tutor.email = str(data["email"]).lower()

    if "full_name" in data and data["full_name"] is not None:
        tutor.full_name = data["full_name"].strip()

    if "phone" in data:
        if data["phone"]:
            tutor.phone = normalize_phone(data["phone"])
        else:
            tutor.phone = None

    if "is_active" in data and data["is_active"] is not None:
        new_active = bool(data["is_active"])
        if tutor.is_active and not new_active:
            _park_teacher_assignments(tutor)
        elif not tutor.is_active and new_active:
            _restore_teacher_assignments(tutor)
        tutor.is_active = new_active

    if "teaching_board" in data:
        tutor.teaching_board = data["teaching_board"]

    tutor.updated_at = datetime.now(UTC)
    db.flush()
    return _teacher_detail(db, tutor)


def delete_teacher(db: Session, actor: User, teacher_id: int) -> None:
    tutor = require_user_by_account_id(db, teacher_id)
    assert_actor_manages_teacher(actor, tutor)
    db.delete(tutor)
    db.flush()


def _assigned_class_ids(teaching_classes: list | None) -> set[str]:
    ids: set[str] = set()
    if not isinstance(teaching_classes, list):
        return ids
    for item in teaching_classes:
        if isinstance(item, dict) and item.get("school_class_id") is not None:
            ids.add(str(item["school_class_id"]))
    return ids


def _merge_school_classes(
    teaching_classes: list | None,
    school_classes: list[SchoolClass],
) -> list[dict]:
    merged: list[dict] = []
    if isinstance(teaching_classes, list):
        for item in teaching_classes:
            if isinstance(item, dict):
                merged.append(dict(item))
    seen = _assigned_class_ids(merged)
    for school_class in school_classes:
        class_key = str(school_class.seq)
        if class_key in seen:
            continue
        merged.append(
            {
                "school_class_id": school_class.seq,
                "grade": school_class.grade,
                "sections": [school_class.section],
                "curriculum": school_class.curriculum,
            }
        )
        seen.add(class_key)
    return merged


def _merge_subject_names(existing: list[str] | None, names: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in list(existing or []) + names:
        label = str(value).strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        merged.append(label)
    return merged


def assign_classes_to_teachers(
    db: Session,
    actor: User,
    teacher_ids: list[int],
    class_ids: list[int],
    *,
    school_id: int | None = None,
) -> TeacherAssignResponse:
    target_school = resolve_target_school_id(actor, school_id=school_id)
    unique_class_ids = list(dict.fromkeys(class_ids))
    school_classes = list(
        db.scalars(
            select(SchoolClass).where(
                SchoolClass.school_id == target_school,
                SchoolClass.seq.in_(unique_class_ids),
            )
        )
    )
    if len(school_classes) != len(unique_class_ids):
        raise AuthException("One or more classes are invalid for this school.", status.HTTP_400_BAD_REQUEST)

    updated = 0
    notified: list[int] = []
    for teacher_id in dict.fromkeys(teacher_ids):
        tutor = require_user_by_account_id(db, teacher_id)
        assert_actor_manages_teacher(actor, tutor)
        tutor.teaching_classes = _merge_school_classes(tutor.teaching_classes, school_classes)
        # Keep teaching_board as curriculum (never a subject name).
        if school_classes:
            tutor.teaching_board = school_classes[0].curriculum
        tutor.updated_at = datetime.now(UTC)
        updated += 1
        notified.append(tutor.id)

    db.flush()
    if notified:
        labels = ", ".join(f"{c.grade}-{c.section}" for c in school_classes[:5])
        more = f" (+{len(school_classes) - 5} more)" if len(school_classes) > 5 else ""
        notify_users(
            db,
            recipient_ids=notified,
            type="classes_assigned",
            title="Classes assigned",
            body=f"You were assigned class(es): {labels}{more}.",
            actor_id=actor.id,
            school_id=target_school,
            link="/tutor/dashboard",
        )
    return TeacherAssignResponse(
        updated=updated,
        message=f"Classes assigned to {updated} teacher(s).",
    )


def assign_subjects_to_teachers(
    db: Session,
    actor: User,
    teacher_ids: list[int],
    subject_ids: list[int],
    *,
    school_id: int | None = None,
) -> TeacherAssignResponse:
    target_school = resolve_target_school_id(actor, school_id=school_id)
    unique_subject_ids = list(dict.fromkeys(subject_ids))
    subjects = list(
        db.scalars(
            select(SchoolSubject).where(
                SchoolSubject.school_id == target_school,
                SchoolSubject.seq.in_(unique_subject_ids),
            )
        )
    )
    if len(subjects) != len(unique_subject_ids):
        raise AuthException("One or more subjects are invalid for this school.", status.HTTP_400_BAD_REQUEST)

    subject_names = [s.name for s in subjects]
    updated = 0
    notified: list[int] = []
    for teacher_id in dict.fromkeys(teacher_ids):
        tutor = require_user_by_account_id(db, teacher_id)
        assert_actor_manages_teacher(actor, tutor)
        tutor.teaching_subjects = _merge_subject_names(tutor.teaching_subjects, subject_names) or None
        tutor.updated_at = datetime.now(UTC)
        updated += 1
        notified.append(tutor.id)

    db.flush()
    if notified:
        labels = ", ".join(subject_names[:5])
        more = f" (+{len(subject_names) - 5} more)" if len(subject_names) > 5 else ""
        notify_users(
            db,
            recipient_ids=notified,
            type="subjects_assigned",
            title="Subjects assigned",
            body=f"You were assigned subject(s): {labels}{more}.",
            actor_id=actor.id,
            school_id=target_school,
            link="/tutor/dashboard",
        )
    return TeacherAssignResponse(
        updated=updated,
        message=f"Subjects assigned to {updated} teacher(s).",
    )


def teacher_options() -> TeacherOptionsResponse:
    return TeacherOptionsResponse(
        subjects=[m.value for m in TeachingSubjectEnum],
        default_limit=DEFAULT_PAGE_LIMIT,
        page_size_options=list(PAGE_SIZE_OPTIONS),
    )


def export_teachers_csv(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
    q: str | None = None,
    subject: str | None = None,
    status_filter: str | None = None,
) -> str:
    result = list_teachers(
        db,
        actor,
        school_id=school_id,
        q=q,
        subject=subject,
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
            "subject",
            "grades",
            "assigned_students",
            "status",
            "last_login",
        ]
    )
    for t in result.items:
        writer.writerow(
            [
                t.user_id,
                t.full_name,
                t.email,
                t.phone or "",
                t.subject or "",
                t.grades or "",
                t.assigned_students,
                "Active" if t.is_active else "Inactive",
                t.last_login.isoformat() if t.last_login else "",
            ]
        )
    return buf.getvalue()


if __name__ == "__main__":
    merged = _merge_school_classes(
        [{"grade": "9", "sections": ["A"]}],
        [],  # type: ignore[arg-type]
    )
    assert merged == [{"grade": "9", "sections": ["A"]}]
    names = _merge_subject_names(["Math"], ["Science", "English"])
    assert names == ["Math", "Science", "English"]
    fake = type("T", (), {"teaching_classes": [{"grade": "9", "sections": ["A"], "curriculum": "CBSE"}]})()
    assert _grades_label(fake) == "9-A · CBSE"
    assert _grade_section_labels({"grade": "10", "sections": ["A"], "curriculum": "CBSE"}) == ["10-A · CBSE"]
    assert _grade_section_labels({"grade": "9", "sections": ["A", "B"], "curriculum": "CBSE"}) == [
        "9-A · CBSE",
        "9-B · CBSE",
    ]

    parked = type(
        "U",
        (),
        {
            "teaching_subjects": ["Math"],
            "teaching_board": "Math",
            "teaching_classes": [{"grade": "9", "sections": ["A"], "school_class_id": 1}],
            "excluded_student_ids": [12],
            "assignment_snapshot": None,
        },
    )()
    _park_teacher_assignments(parked)
    assert parked.teaching_subjects is None and parked.teaching_classes is None
    assert parked.assignment_snapshot["teaching_subjects"] == ["Math"]
    assert parked.assignment_snapshot["excluded_student_ids"] == [12]
    _restore_teacher_assignments(parked)
    assert parked.teaching_subjects == ["Math"]
    assert parked.excluded_student_ids == [12]
    assert parked.assignment_snapshot is None
    print("teachers service self-check ok")


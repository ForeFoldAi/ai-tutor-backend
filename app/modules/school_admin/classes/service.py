from __future__ import annotations

from datetime import UTC, datetime

from fastapi import status
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.school_admin.classes.constants import (
    DEFAULT_PAGE_LIMIT,
    MAX_BULK_IMPORT,
    MAX_PAGE_LIMIT,
    PAGE_SIZE_OPTIONS,
)
from app.modules.school_admin.classes.models import SchoolClass, SchoolClassSubject, SchoolSubject
from app.modules.school_admin.classes.schemas import (
    BulkRowError,
    ClassBulkCreateResponse,
    ClassCreateItem,
    ClassListResponse,
    ClassOptionsResponse,
    ClassResponse,
    ClassSubjectMappingSaveRequest,
    ClassSubjectMappingsResponse,
    ClassUpdateRequest,
    PaginationMeta,
    SubjectBulkCreateResponse,
    SubjectCreateItem,
    SubjectListResponse,
    SubjectResponse,
    SubjectUpdateRequest,
)
from app.modules.school_admin.classes.validators import (
    ClassScope,
    assert_actor_manages_class,
    assert_actor_manages_subject,
    assert_can_manage_classes,
    assert_curriculum_allowed,
    assert_curriculum_in_scope,
    is_individual_tutor,
    resolve_class_scope,
    scope_curricula,
    sync_individual_tutor_teaching_profile,
)
from app.modules.schools.models import School
from app.modules.users.models import User


def _scrub_subject_name_from_teachers(db: Session, school_id: int, old_name: str, new_name: str | None = None) -> None:
    """Remove or rename a subject name on all tutors for this school (teaching_subjects JSON)."""
    old_key = old_name.strip().lower()
    if not old_key:
        return
    tutors = list(
        db.scalars(select(User).where(User.school_id == school_id, User.role == Role.TUTOR))
    )
    for tutor in tutors:
        raw = tutor.teaching_subjects
        if isinstance(raw, list) and raw:
            next_list: list[str] = []
            changed = False
            for item in raw:
                label = str(item).strip()
                if not label:
                    continue
                if label.lower() == old_key:
                    changed = True
                    if new_name and new_name.strip():
                        next_list.append(new_name.strip())
                    continue
                next_list.append(label)
            if changed:
                # de-dupe preserving order
                seen: set[str] = set()
                deduped: list[str] = []
                for label in next_list:
                    k = label.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    deduped.append(label)
                tutor.teaching_subjects = deduped or None

        snap = tutor.assignment_snapshot
        if isinstance(snap, dict):
            snap_subjects = snap.get("teaching_subjects")
            if isinstance(snap_subjects, list) and snap_subjects:
                next_snap: list[str] = []
                snap_changed = False
                for item in snap_subjects:
                    label = str(item).strip()
                    if not label:
                        continue
                    if label.lower() == old_key:
                        snap_changed = True
                        if new_name and new_name.strip():
                            next_snap.append(new_name.strip())
                        continue
                    next_snap.append(label)
                if snap_changed:
                    seen_s: set[str] = set()
                    deduped_s: list[str] = []
                    for label in next_snap:
                        k = label.lower()
                        if k in seen_s:
                            continue
                        seen_s.add(k)
                        deduped_s.append(label)
                    snap = {**snap, "teaching_subjects": deduped_s or None}
                    tutor.assignment_snapshot = snap


def _next_subject_seq(db: Session, scope: ClassScope) -> int:
    current = db.scalar(
        select(func.coalesce(func.max(SchoolSubject.seq), 0)).where(scope.subject_filter())
    )
    return int(current or 0) + 1


def _next_class_seq(db: Session, scope: ClassScope) -> int:
    current = db.scalar(
        select(func.coalesce(func.max(SchoolClass.seq), 0)).where(scope.class_filter())
    )
    return int(current or 0) + 1


def _validate_curriculum(db: Session, actor: User, scope: ClassScope, curriculum: str) -> str:
    if scope.school_id is not None:
        school = db.get(School, scope.school_id)
        if not school:
            raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
        return assert_curriculum_allowed(school, curriculum)
    return assert_curriculum_in_scope(scope_curricula(db, actor, scope), curriculum)


def _maybe_sync_tutor(db: Session, actor: User) -> None:
    if is_individual_tutor(actor):
        sync_individual_tutor_teaching_profile(db, actor)


def _subject_response(subject: SchoolSubject) -> SubjectResponse:
    return SubjectResponse(
        id=subject.seq,
        name=subject.name,
        code=subject.code,
        is_active=subject.is_active,
        created_at=subject.created_at,
        updated_at=subject.updated_at,
    )


def _subject_by_seq(db: Session, scope: ClassScope, subject_seq: int) -> SchoolSubject | None:
    return db.scalar(
        select(SchoolSubject).where(
            scope.subject_filter(),
            SchoolSubject.seq == subject_seq,
        )
    )


def _class_by_seq(db: Session, scope: ClassScope, class_seq: int) -> SchoolClass | None:
    return db.scalar(
        select(SchoolClass).where(
            scope.class_filter(),
            SchoolClass.seq == class_seq,
        )
    )


def _class_response(school_class: SchoolClass, *, students: int = 0, teachers: int = 0) -> ClassResponse:
    return ClassResponse(
        id=school_class.seq,
        grade=school_class.grade,
        section=school_class.section,
        curriculum=school_class.curriculum,
        students=students,
        teachers=teachers,
        created_at=school_class.created_at,
        updated_at=school_class.updated_at,
    )


def _user_matches_class(user: User, school_class: SchoolClass) -> bool:
    """True if user's teaching_classes overlap this grade/section (and curriculum when set)."""
    raw = user.teaching_classes
    if not isinstance(raw, list) or not raw:
        return False
    target_section = school_class.section.strip().upper()
    target_curriculum = school_class.curriculum.strip().lower()
    for item in raw:
        if not isinstance(item, dict):
            continue
        sc_id = item.get("school_class_id")
        if sc_id is not None and str(sc_id).strip().lstrip("-").isdigit():
            if int(sc_id) == school_class.seq:
                return True
        grade = str(item.get("grade", "")).strip()
        if grade != school_class.grade:
            continue
        sections = {
            str(s).strip().upper()
            for s in (item.get("sections") or [])
            if str(s).strip()
        }
        if target_section not in sections:
            continue
        curriculum = str(item.get("curriculum", "") or "").strip().lower()
        if curriculum and curriculum != target_curriculum:
            continue
        return True
    return False


def _roster_counts_for_classes(
    db: Session,
    scope: ClassScope,
    classes: list[SchoolClass],
) -> dict[int, tuple[int, int]]:
    """seq → (students, teachers)."""
    if not classes:
        return {}
    out: dict[int, tuple[int, int]] = {c.seq: (0, 0) for c in classes}

    if scope.school_id is not None:
        people = list(
            db.scalars(
                select(User).where(
                    User.school_id == scope.school_id,
                    User.role.in_([Role.STUDENT, Role.TUTOR]),
                    User.is_active.is_(True),
                )
            )
        )
        for school_class in classes:
            students = 0
            teachers = 0
            for user in people:
                if not _user_matches_class(user, school_class):
                    continue
                if user.role == Role.STUDENT:
                    students += 1
                elif user.role == Role.TUTOR:
                    teachers += 1
            out[school_class.seq] = (students, teachers)
        return out

    # Individual tutor: students they created; teacher is the owner (0 or 1).
    students_pool = list(
        db.scalars(
            select(User).where(
                User.created_by == scope.owner_user_id,
                User.role == Role.STUDENT,
                User.is_active.is_(True),
            )
        )
    )
    owner = db.get(User, scope.owner_user_id) if scope.owner_user_id else None
    for school_class in classes:
        students = sum(1 for u in students_pool if _user_matches_class(u, school_class))
        teachers = 1 if owner and _user_matches_class(owner, school_class) else 0
        out[school_class.seq] = (students, teachers)
    return out


def class_options(db: Session, actor: User, *, school_id: int | None = None) -> ClassOptionsResponse:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor, school_id=school_id)
    return ClassOptionsResponse(
        curricula=scope_curricula(db, actor, scope),
        default_limit=DEFAULT_PAGE_LIMIT,
        page_size_options=list(PAGE_SIZE_OPTIONS),
    )


def list_subjects(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> SubjectListResponse:
    assert_can_manage_classes(actor)
    offset = max(0, offset)
    scope = resolve_class_scope(actor, school_id=school_id)

    stmt = select(SchoolSubject).where(scope.subject_filter())
    total = int(db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
    ordered = stmt.order_by(SchoolSubject.seq.asc()).offset(offset)
    if limit is not None:
        ordered = ordered.limit(min(max(1, limit), MAX_PAGE_LIMIT))
    items = list(db.scalars(ordered))
    effective_limit = total if limit is None else min(max(1, limit), MAX_PAGE_LIMIT)
    return SubjectListResponse(
        items=[_subject_response(s) for s in items],
        meta=PaginationMeta(total=total, limit=effective_limit, offset=offset, default_limit=0),
    )


def bulk_create_subjects(
    db: Session,
    actor: User,
    items: list[SubjectCreateItem],
    *,
    school_id: int | None = None,
) -> SubjectBulkCreateResponse:
    assert_can_manage_classes(actor)
    if len(items) > MAX_BULK_IMPORT:
        raise AuthException(f"At most {MAX_BULK_IMPORT} subjects per request.", status.HTTP_400_BAD_REQUEST)

    scope = resolve_class_scope(actor, school_id=school_id)
    created: list[SubjectResponse] = []
    errors: list[BulkRowError] = []
    seen_codes: set[str] = set()
    row_index = 0

    for item in items:
        name = item.name.strip()
        code_key = item.code.upper()
        if not name and not code_key:
            continue
        row_index += 1
        if not name or not code_key:
            errors.append(BulkRowError(row=row_index, field="name", message="Name and code are required."))
            continue
        if code_key in seen_codes:
            errors.append(BulkRowError(row=row_index, field="code", message="Duplicate code in request."))
            continue
        seen_codes.add(code_key)

        existing = db.scalar(
            select(SchoolSubject).where(
                scope.subject_filter(),
                func.upper(SchoolSubject.code) == code_key,
            )
        )
        if existing:
            errors.append(BulkRowError(row=row_index, field="code", message="Subject code already exists."))
            continue

        try:
            subject = SchoolSubject(
                school_id=scope.school_id,
                owner_user_id=scope.owner_user_id,
                seq=_next_subject_seq(db, scope),
                name=name,
                code=code_key,
                is_active=True,
            )
            db.add(subject)
            db.flush()
            created.append(_subject_response(subject))
        except Exception as exc:  # ponytail: row-level errors without aborting batch
            errors.append(BulkRowError(row=row_index, field="code", message=str(exc)))

    if not created and not errors:
        raise AuthException("No subjects to create.", status.HTTP_400_BAD_REQUEST)

    _maybe_sync_tutor(db, actor)
    return SubjectBulkCreateResponse(created=created, errors=errors)


def list_classes(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
    q: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> ClassListResponse:
    offset = max(0, offset)
    scope = resolve_class_scope(actor, school_id=school_id)

    stmt = select(SchoolClass).where(scope.class_filter())
    if q and q.strip():
        term = f"%{q.strip().lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(SchoolClass.grade).like(term),
                func.lower(SchoolClass.section).like(term),
                func.lower(SchoolClass.curriculum).like(term),
            )
        )
    total = int(db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
    ordered = stmt.order_by(SchoolClass.seq.asc()).offset(offset)
    if limit is not None:
        ordered = ordered.limit(min(max(1, limit), MAX_PAGE_LIMIT))
    items = list(db.scalars(ordered))
    effective_limit = total if limit is None else min(max(1, limit), MAX_PAGE_LIMIT)
    counts = _roster_counts_for_classes(db, scope, items)
    return ClassListResponse(
        items=[
            _class_response(
                c,
                students=counts.get(c.seq, (0, 0))[0],
                teachers=counts.get(c.seq, (0, 0))[1],
            )
            for c in items
        ],
        meta=PaginationMeta(total=total, limit=effective_limit, offset=offset, default_limit=0),
    )


def bulk_create_classes(
    db: Session,
    actor: User,
    items: list[ClassCreateItem],
    *,
    school_id: int | None = None,
) -> ClassBulkCreateResponse:
    assert_can_manage_classes(actor)
    if len(items) > MAX_BULK_IMPORT:
        raise AuthException(f"At most {MAX_BULK_IMPORT} classes per request.", status.HTTP_400_BAD_REQUEST)

    scope = resolve_class_scope(actor, school_id=school_id)
    created: list[ClassResponse] = []
    errors: list[BulkRowError] = []
    seen_keys: set[str] = set()

    for index, item in enumerate(items, start=1):
        try:
            curriculum = _validate_curriculum(db, actor, scope, item.curriculum)
        except AuthException as exc:
            errors.append(BulkRowError(row=index, field="curriculum", message=exc.detail))
            continue

        grade = item.grade.strip()
        section = item.section.strip().upper()
        key = f"{grade}-{section}-{curriculum.lower()}"
        if key in seen_keys:
            errors.append(BulkRowError(row=index, field="grade", message="Duplicate class in request."))
            continue
        seen_keys.add(key)

        existing = db.scalar(
            select(SchoolClass).where(
                scope.class_filter(),
                SchoolClass.grade == grade,
                func.upper(SchoolClass.section) == section,
                func.lower(SchoolClass.curriculum) == curriculum.lower(),
            )
        )
        if existing:
            errors.append(BulkRowError(row=index, field="grade", message="Class already exists."))
            continue

        try:
            school_class = SchoolClass(
                school_id=scope.school_id,
                owner_user_id=scope.owner_user_id,
                seq=_next_class_seq(db, scope),
                grade=grade,
                section=section,
                curriculum=curriculum,
            )
            db.add(school_class)
            db.flush()
            created.append(_class_response(school_class))
        except Exception as exc:
            errors.append(BulkRowError(row=index, field="grade", message=str(exc)))

    _maybe_sync_tutor(db, actor)
    return ClassBulkCreateResponse(created=created, errors=errors)


def get_class_subject_mappings(
    db: Session,
    actor: User,
    *,
    school_id: int | None = None,
) -> ClassSubjectMappingsResponse:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor, school_id=school_id)
    classes = list(
        db.scalars(
            select(SchoolClass)
            .where(scope.class_filter())
            .options(
                selectinload(SchoolClass.subject_links).selectinload(SchoolClassSubject.subject),
            )
        )
    )

    mappings: dict[str, dict[str, bool]] = {}
    for school_class in classes:
        key = f"{school_class.seq}:{school_class.curriculum}"
        mappings[key] = {str(link.subject.seq): True for link in school_class.subject_links}

    return ClassSubjectMappingsResponse(mappings=mappings)


def save_class_subject_mapping(
    db: Session,
    actor: User,
    class_seq: int,
    payload: ClassSubjectMappingSaveRequest,
) -> ClassSubjectMappingsResponse:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor)
    school_class = _class_by_seq(db, scope, class_seq)
    if not school_class:
        raise AuthException("Class not found.", status.HTTP_404_NOT_FOUND)
    assert_actor_manages_class(actor, school_class)

    valid_subject_ids: set[int] = set()
    for subject_seq in payload.subject_ids:
        subject = _subject_by_seq(db, scope, subject_seq)
        if not subject:
            raise AuthException("Invalid subject for this school.", status.HTTP_400_BAD_REQUEST)
        valid_subject_ids.add(subject.id)

    from app.modules.student_learning.enrollment import (
        compute_scope_key,
        enrolled_subject_names,
        students_on_school_class,
    )
    from app.modules.student_learning.models import StudentLearningStreak
    from app.modules.student_learning.service import reset_student_learning

    old_names = enrolled_subject_names(db, school_class)
    old_scope = compute_scope_key(school_class, old_names)

    db.execute(delete(SchoolClassSubject).where(SchoolClassSubject.school_class_id == school_class.id))
    for subject_id in valid_subject_ids:
        db.add(SchoolClassSubject(school_class_id=school_class.id, subject_id=subject_id))
    school_class.updated_at = datetime.now(UTC)
    db.flush()

    new_names = enrolled_subject_names(db, school_class)
    new_scope = compute_scope_key(school_class, new_names)
    if new_scope != old_scope:
        for student in students_on_school_class(db, school_class):
            streak = db.scalar(
                select(StudentLearningStreak).where(StudentLearningStreak.user_id == student.id)
            )
            # Always reset when class subject map changes (stats restart)
            if streak is None or streak.scope_key != new_scope or old_scope != new_scope:
                reset_student_learning(db, student.id)

    _maybe_sync_tutor(db, actor)
    return get_class_subject_mappings(db, actor, school_id=scope.school_id)


def delete_class(db: Session, actor: User, class_seq: int) -> None:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor)
    school_class = _class_by_seq(db, scope, class_seq)
    if not school_class:
        raise AuthException("Class not found.", status.HTTP_404_NOT_FOUND)
    assert_actor_manages_class(actor, school_class)
    db.delete(school_class)
    db.flush()
    _maybe_sync_tutor(db, actor)


def update_class(db: Session, actor: User, class_seq: int, payload: ClassUpdateRequest) -> ClassResponse:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor)
    school_class = _class_by_seq(db, scope, class_seq)
    if not school_class:
        raise AuthException("Class not found.", status.HTTP_404_NOT_FOUND)
    assert_actor_manages_class(actor, school_class)

    grade = payload.grade.strip()
    section = payload.section.strip().upper()
    curriculum = _validate_curriculum(db, actor, scope, payload.curriculum)
    clash = db.scalar(
        select(SchoolClass).where(
            scope.class_filter(),
            SchoolClass.grade == grade,
            func.upper(SchoolClass.section) == section,
            func.lower(SchoolClass.curriculum) == curriculum.lower(),
            SchoolClass.id != school_class.id,
        )
    )
    if clash:
        raise AuthException("Class already exists.", status.HTTP_409_CONFLICT)

    school_class.grade = grade
    school_class.section = section
    school_class.curriculum = curriculum
    school_class.updated_at = datetime.now(UTC)
    db.flush()
    _maybe_sync_tutor(db, actor)
    counts = _roster_counts_for_classes(db, scope, [school_class])
    students, teachers = counts.get(school_class.seq, (0, 0))
    return _class_response(school_class, students=students, teachers=teachers)


def delete_subject(db: Session, actor: User, subject_seq: int) -> None:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor)
    subject = _subject_by_seq(db, scope, subject_seq)
    if not subject:
        raise AuthException("Subject not found.", status.HTTP_404_NOT_FOUND)
    assert_actor_manages_subject(actor, subject)
    if scope.school_id is not None:
        _scrub_subject_name_from_teachers(db, scope.school_id, subject.name)
    db.delete(subject)
    db.flush()
    _maybe_sync_tutor(db, actor)


def update_subject(db: Session, actor: User, subject_seq: int, payload: SubjectUpdateRequest) -> SubjectResponse:
    assert_can_manage_classes(actor)
    scope = resolve_class_scope(actor)
    subject = _subject_by_seq(db, scope, subject_seq)
    if not subject:
        raise AuthException("Subject not found.", status.HTTP_404_NOT_FOUND)
    assert_actor_manages_subject(actor, subject)

    name = payload.name.strip()
    code = payload.code.strip().upper()
    clash = db.scalar(
        select(SchoolSubject).where(
            scope.subject_filter(),
            func.upper(SchoolSubject.code) == code,
            SchoolSubject.id != subject.id,
        )
    )
    if clash:
        raise AuthException("Subject code already exists.", status.HTTP_409_CONFLICT)

    old_name = subject.name
    subject.name = name
    subject.code = code
    subject.updated_at = datetime.now(UTC)
    if scope.school_id is not None and old_name.strip().lower() != name.lower():
        _scrub_subject_name_from_teachers(db, scope.school_id, old_name, new_name=name)
    db.flush()
    _maybe_sync_tutor(db, actor)
    return _subject_response(subject)

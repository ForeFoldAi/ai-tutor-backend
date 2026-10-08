"""Resolve a student's enrolled subjects from SchoolClassSubject (school + individual tutor)."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.modules.auth.constants import Role
from app.modules.catalog.models import BoardEnum, ClassEnum, SyllabusSubject, Textbook, TextbookUpload
from app.modules.catalog.publisher_books import filter_uploads_for_book, publisher_book_of
from app.modules.catalog.schemas import StudentChapterResponse, StudentSubjectResponse
from app.modules.school_admin.classes.models import SchoolClass, SchoolClassSubject, SchoolSubject
from app.modules.users.models import User
from app.modules.auth.signup.models import UserSignupProfile


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _parse_class_level(raw: str | None) -> ClassEnum | None:
    if not raw or not str(raw).strip():
        return None
    grade_str = str(raw).strip()
    canon = f"CLASS_{grade_str}" if grade_str.isdigit() else grade_str
    try:
        return ClassEnum(canon)
    except ValueError:
        return None


def _parse_board(raw: str | None) -> BoardEnum | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    try:
        return BoardEnum(s)
    except ValueError:
        pass
    aliases = {
        "state board": BoardEnum.STATE_BOARD,
        "state_board": BoardEnum.STATE_BOARD,
        "cambridge": BoardEnum.CAMBRIDGE,
        "igcse": BoardEnum.CAMBRIDGE,
    }
    return aliases.get(s.lower())


def _chapter_sort_key(tb: TextbookUpload) -> tuple[int, str, int, str]:
    f = tb.file_name or ""
    c = tb.chapter or ""
    m = re.search(r"(?:chapter|unit|ch)\s*[-_.]?\s*(\d+)(?:[.\s_-]*([a-zA-Z]))?\b", f, re.I)
    if not m:
        m = re.search(r"(?:chapter|unit|ch)\s*[-_.]?\s*(\d+)(?:[.\s_-]+([a-zA-Z])\b(?![a-z]))?", c, re.I)
    if m:
        num = int(m.group(1))
        sub = (m.group(2) or "").upper()
        if not sub:
            rm = re.search(r"(?:reading|part|section)\s+([a-zA-Z])\b", c, re.I)
            if rm:
                sub = rm.group(1).upper()
        is_raw_pdf = 1 if f.lower().endswith(".pdf") else 0
        return (num, sub, is_raw_pdf, c)
    return (9999, "", 1, c)


def _subjects_from_catalog(
    db: Session,
    *,
    board: BoardEnum,
    class_level: ClassEnum,
    subject_names: list[str],
    book_by_subject: dict[str, int] | None = None,
) -> list[StudentSubjectResponse]:
    allowed = {_norm(n) for n in subject_names}
    if not allowed:
        return []

    subjects = list(
        db.scalars(
            select(SyllabusSubject)
            .where(SyllabusSubject.board == board, SyllabusSubject.class_level == class_level)
            .order_by(SyllabusSubject.subject_name)
        )
    )
    subjects = [s for s in subjects if _norm(s.subject_name) in allowed]

    textbooks = list(
        db.scalars(
            select(TextbookUpload)
            .where(TextbookUpload.board == board, TextbookUpload.class_level == class_level)
            .order_by(TextbookUpload.chapter)
        )
    )
    tb_by_subject: dict[str, list[TextbookUpload]] = {}
    for tb in textbooks:
        key = _norm(tb.subject_name)
        if key not in allowed:
            continue
        tb_by_subject.setdefault(key, []).append(tb)

    book_of = publisher_book_of(db, (tb.id for rows in tb_by_subject.values() for tb in rows))
    chosen = book_by_subject or {}
    for k in tb_by_subject:
        tb_by_subject[k] = filter_uploads_for_book(tb_by_subject[k], book_of, chosen.get(k))
        tb_by_subject[k].sort(key=_chapter_sort_key)

    result: list[StudentSubjectResponse] = []
    seen: set[str] = set()
    for subj in subjects:
        key = _norm(subj.subject_name)
        seen.add(key)
        chapters = [
            StudentChapterResponse(id=tb.id, chapter=tb.chapter, file_name=tb.file_name)
            for tb in tb_by_subject.get(key, [])
        ]
        result.append(
            StudentSubjectResponse(
                id=subj.id,
                board=board,
                class_level=class_level,
                subject_name=subj.subject_name,
                chapters=chapters,
            )
        )

    for name in subject_names:
        key = _norm(name)
        if key in seen:
            continue
        chapters = [
            StudentChapterResponse(id=tb.id, chapter=tb.chapter, file_name=tb.file_name)
            for tb in tb_by_subject.get(key, [])
        ]
        result.append(
            StudentSubjectResponse(
                id=-(abs(hash(key)) % 1_000_000_000) or -1,
                board=board,
                class_level=class_level,
                subject_name=name,
                chapters=chapters,
            )
        )

    result.sort(key=lambda s: s.subject_name.lower())
    return result


def student_class_entry(user: User) -> dict | None:
    grades = user.teaching_classes or []
    if not grades or not isinstance(grades[0], dict):
        return None
    return grades[0]


def resolve_student_school_class(db: Session, user: User) -> SchoolClass | None:
    """Resolve SchoolClass from teaching_classes (seq stored as school_class_id)."""
    entry = student_class_entry(user)
    if not entry:
        return None

    seq_raw = entry.get("school_class_id")
    grade = str(entry.get("grade", "")).strip()
    sections = entry.get("sections") or []
    section = str(sections[0]).strip().upper() if sections else ""
    curriculum = str(entry.get("curriculum") or user.teaching_board or "").strip()

    # Prefer explicit seq within school or owning tutor scope
    if seq_raw is not None and str(seq_raw).strip().lstrip("-").isdigit():
        seq = int(seq_raw)
        if user.school_id is not None:
            row = db.scalar(
                select(SchoolClass).where(
                    SchoolClass.school_id == user.school_id,
                    SchoolClass.seq == seq,
                )
            )
            if row:
                return row
        if user.created_by is not None:
            row = db.scalar(
                select(SchoolClass).where(
                    SchoolClass.owner_user_id == user.created_by,
                    SchoolClass.school_id.is_(None),
                    SchoolClass.seq == seq,
                )
            )
            if row:
                return row

    # Fallback: grade + section (+ curriculum) match
    if not grade or not section:
        return None

    q = select(SchoolClass).where(
        SchoolClass.grade == grade,
        SchoolClass.section == section,
    )
    if user.school_id is not None:
        q = q.where(SchoolClass.school_id == user.school_id)
    elif user.created_by is not None:
        q = q.where(
            SchoolClass.owner_user_id == user.created_by,
            SchoolClass.school_id.is_(None),
        )
    else:
        return None

    if curriculum:
        rows = list(db.scalars(q))
        for row in rows:
            if row.curriculum.strip().lower() == curriculum.lower():
                return row
        return rows[0] if rows else None

    return db.scalar(q)


def enrolled_subject_names(db: Session, school_class: SchoolClass) -> list[str]:
    links = list(
        db.scalars(
            select(SchoolClassSubject)
            .where(SchoolClassSubject.school_class_id == school_class.id)
            .options(selectinload(SchoolClassSubject.subject))
        )
    )
    names: list[str] = []
    seen: set[str] = set()
    for link in links:
        subj: SchoolSubject | None = link.subject
        if not subj or not subj.is_active:
            continue
        key = _norm(subj.name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(subj.name.strip())
    return sorted(names, key=lambda n: n.lower())


def class_book_choices(db: Session, school_class: SchoolClass) -> dict[str, int]:
    """Normalised subject name -> non-default textbook id chosen for this class."""
    rows = db.execute(
        select(SchoolSubject.name, Textbook.id)
        .join(SchoolClassSubject, SchoolClassSubject.subject_id == SchoolSubject.id)
        .join(Textbook, Textbook.id == SchoolClassSubject.textbook_id)
        .where(SchoolClassSubject.school_class_id == school_class.id, Textbook.is_default.is_(False))
    )
    return {_norm(name): int(tid) for name, tid in rows}


def student_book_choices(db: Session, user: User, school_class: SchoolClass | None) -> dict[str, int]:
    """Class picks, overridden per subject by user.publisher_choices (0 = default pool; deleted books ignored)."""
    books = class_book_choices(db, school_class) if school_class is not None else {}
    own = {k: int(v) for k, v in (user.publisher_choices or {}).items() if v is not None}
    ids = [v for v in own.values() if v]
    valid = set(db.scalars(select(Textbook.id).where(Textbook.id.in_(ids), Textbook.is_default.is_(False)))) if ids else set()
    for subject, book_id in own.items():
        if book_id == 0:
            books.pop(subject, None)
        elif book_id in valid:
            books[subject] = book_id
    return books


def _is_individual_student(user: User) -> bool:
    # Individual student signup: no school scope, no tutor owner scope.
    return user.school_id is None and user.created_by is None


def favorite_subject_names(db: Session, user: User) -> list[str]:
    """Individual student's chosen subjects (from signup profile)."""
    if not _is_individual_student(user):
        return []
    profile = db.get(UserSignupProfile, user.id)
    if not profile or not profile.favorite_subjects:
        return []
    names = [s.value for s in profile.favorite_subjects if getattr(s, "value", None)]
    return sorted(names, key=lambda n: n.lower())


def effective_subject_names(
    db: Session,
    *,
    user: User,
    school_class: SchoolClass,
    mapped_names: list[str],
) -> list[str]:
    """
    For individual students, prefer their favorite subjects.
    For tagged students, keep the class-mapped subjects stable.
    """
    fav = favorite_subject_names(db, user)
    return fav or mapped_names


def individual_board_and_class(
    db: Session,
    user: User,
) -> tuple[BoardEnum | None, ClassEnum | None]:
    """Resolve board + class from signup profile / user.teaching_* for individual students."""
    profile = db.get(UserSignupProfile, user.id)
    board_raw: str | None = None
    grade_raw: str | None = None

    if profile:
        if profile.curricula:
            board_raw = profile.curricula[0].value
        if profile.student_grade is not None:
            grade_raw = profile.student_grade.value

    entry = student_class_entry(user)
    if entry:
        grade_raw = grade_raw or str(entry.get("grade", "")).strip() or None
        cur = str(entry.get("curriculum") or "").strip()
        board_raw = board_raw or (cur if cur else None)

    if not board_raw and user.teaching_board:
        board_raw = str(user.teaching_board).strip()

    return _parse_board(board_raw), _parse_class_level(grade_raw)


def _with_books(key: str, normed: list[str], books: dict[str, int] | None) -> str:
    # Only appended when a publisher book is chosen, so existing keys (and student progress) stay unchanged.
    picked = [f"{n}={books[n]}" for n in normed if books and n in books]
    return f"{key}|books:{','.join(picked)}" if picked else key


def individual_scope_key(
    user: User,
    board: BoardEnum,
    class_level: ClassEnum,
    subject_names: list[str],
    books: dict[str, int] | None = None,
) -> str:
    normed = sorted(_norm(n) for n in subject_names)
    return _with_books(f"ind:{user.id}:{board.value}:{class_level.value}:{','.join(normed)}", normed, books)


def compute_scope_key(
    school_class: SchoolClass | None,
    subject_names: list[str],
    books: dict[str, int] | None = None,
) -> str:
    if school_class is None:
        return "none:"
    normed = sorted(_norm(n) for n in subject_names)
    return _with_books(f"{school_class.id}:{','.join(normed)}", normed, books)


def individual_learning_scope(
    db: Session,
    user: User,
) -> tuple[list[str], str]:
    subject_names = favorite_subject_names(db, user)
    board, class_level = individual_board_and_class(db, user)
    if not subject_names or not board or not class_level:
        return [], "none:"
    books = student_book_choices(db, user, None)
    return subject_names, individual_scope_key(user, board, class_level, subject_names, books)


def list_enrolled_subjects(
    db: Session,
    user: User,
    *,
    board: BoardEnum | None = None,
    class_level: ClassEnum | None = None,
) -> tuple[list[StudentSubjectResponse], str]:
    """
    Subjects mapped to the student's class, intersected with syllabus + textbooks.
    Returns (subjects, scope_key). Empty when no class / no mapped subjects.
    """
    if _is_individual_student(user):
        subject_names = favorite_subject_names(db, user)
        if not subject_names:
            return [], "none:"

        if board is None or class_level is None:
            resolved_board, resolved_class = individual_board_and_class(db, user)
            board = board or resolved_board
            class_level = class_level or resolved_class
        if not board or not class_level:
            return [], "none:"

        books = student_book_choices(db, user, None)
        scope_key = individual_scope_key(user, board, class_level, subject_names, books)
        return (
            _subjects_from_catalog(
                db,
                board=board,
                class_level=class_level,
                subject_names=subject_names,
                book_by_subject=books,
            ),
            scope_key,
        )

    school_class = resolve_student_school_class(db, user)
    if school_class is None:
        return [], "none:"

    mapped_names = enrolled_subject_names(db, school_class)
    subject_names = effective_subject_names(
        db, user=user, school_class=school_class, mapped_names=mapped_names
    )
    books = student_book_choices(db, user, school_class)
    scope_key = compute_scope_key(school_class, subject_names, books)
    if not subject_names:
        return [], scope_key

    if board is None:
        board = _parse_board(user.teaching_board) or _parse_board(school_class.curriculum)
    if class_level is None:
        entry = student_class_entry(user)
        grade = entry.get("grade", "") if entry else school_class.grade
        class_level = _parse_class_level(str(grade))

    if not board or not class_level:
        return [], scope_key

    return (
        _subjects_from_catalog(
            db,
            board=board,
            class_level=class_level,
            subject_names=subject_names,
            book_by_subject=books,
        ),
        scope_key,
    )


def students_on_school_class(db: Session, school_class: SchoolClass) -> list[User]:
    """Students whose teaching_classes point at this class (by seq)."""
    seq = school_class.seq
    if school_class.school_id is not None:
        candidates = list(
            db.scalars(
                select(User).where(
                    User.role == Role.STUDENT,
                    User.school_id == school_class.school_id,
                )
            )
        )
    elif school_class.owner_user_id is not None:
        candidates = list(
            db.scalars(
                select(User).where(
                    User.role == Role.STUDENT,
                    User.created_by == school_class.owner_user_id,
                    User.school_id.is_(None),
                )
            )
        )
    else:
        return []

    matched: list[User] = []
    for student in candidates:
        entry = student_class_entry(student)
        if not entry:
            continue
        sid = entry.get("school_class_id")
        if sid is not None and str(sid).strip().lstrip("-").isdigit() and int(sid) == seq:
            matched.append(student)
            continue
        grade = str(entry.get("grade", "")).strip()
        sections = entry.get("sections") or []
        section = str(sections[0]).strip().upper() if sections else ""
        if (
            grade == school_class.grade
            and section == school_class.section
            and str(entry.get("curriculum") or student.teaching_board or "").strip().lower()
            == school_class.curriculum.strip().lower()
        ):
            matched.append(student)
    return matched

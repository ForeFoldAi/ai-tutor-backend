from __future__ import annotations

import csv
import io
import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.public_ids import require_user_by_account_id
from app.modules.auth.service import get_or_create_user_settings, list_tutor_assigned_students
from app.modules.school_admin.classes.models import SchoolClass, SchoolClassSubject
from app.modules.school_admin.students.service import _class_fields
from app.modules.sessions.models import SessionToken
from app.modules.student_learning.models import StudentChapterProgress, StudentStudySession
from app.modules.teacher.assignments.constants import AssignmentStatus, SubmissionStatus
from app.modules.teacher.assignments.models import AssignmentSubmission, TeacherAssignment
from app.modules.teacher.students.constants import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    PAGE_SIZE_OPTIONS,
    RISK_HIGH,
    RISK_LEVELS,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_NOT_STARTED,
)
from app.modules.teacher.students.schemas import (
    ActivityItem,
    AssignmentItem,
    PaginationMeta,
    QuizItem,
    TutorStudentListResponse,
    TutorStudentOptionsResponse,
    TutorStudentProfileResponse,
    TutorStudentRow,
)
from app.modules.users.models import User

IST = ZoneInfo("Asia/Kolkata")

_DONE_SUBMISSION = frozenset({SubmissionStatus.SUBMITTED.value, SubmissionStatus.GRADED.value})


def _require_school_tutor(tutor: User) -> None:
    if tutor.role != Role.TUTOR:
        raise AuthException("Only tutors can access assigned students.", status.HTTP_403_FORBIDDEN)


def _slugify(name: str) -> str:
    slug = re.sub(r"\s+", "-", name.strip().lower())
    return re.sub(r"[^a-z0-9\-]", "", slug) or "student"


def _username_for(db: Session, user: User) -> str:
    settings = get_or_create_user_settings(db, user)
    if settings.username:
        return settings.username
    return user.email.split("@", 1)[0]


def _grade_label(grade: str | None, section: str | None, curriculum: str | None) -> str:
    parts = [p for p in (grade, section, curriculum) if p]
    return " · ".join(parts) if parts else "—"


def risk_from_completion(completion: int) -> str:
    if completion >= 70:
        return RISK_LOW
    if completion >= 45:
        return RISK_MEDIUM
    return RISK_HIGH


def risk_for_student(*, completion: int, has_started: bool = True) -> str:
    """Risk column value. 0% / no learning progress → Not Started (not High)."""
    if not has_started or completion <= 0:
        return RISK_NOT_STARTED
    return risk_from_completion(completion)


def _normalize_risk_label(level: str | None) -> str | None:
    if not level or not str(level).strip():
        return None
    key = str(level).strip().lower().replace("_", " ")
    return {
        "not started": RISK_NOT_STARTED,
        "low": RISK_LOW,
        "medium": RISK_MEDIUM,
        "high": RISK_HIGH,
    }.get(key)


def _format_last_active(last_login: datetime | None) -> str | None:
    if not last_login:
        return None
    return last_login.astimezone(IST).strftime("%d %b %Y, %I:%M %p")


def _last_login_map(db: Session, user_ids: list[int]) -> dict[int, datetime]:
    if not user_ids:
        return {}
    rows = db.execute(
        select(SessionToken.user_id, func.max(SessionToken.created_at))
        .where(SessionToken.user_id.in_(user_ids))
        .group_by(SessionToken.user_id)
    ).all()
    return {int(uid): ts for uid, ts in rows if ts is not None}


def _school_class_for_student(db: Session, student: User) -> SchoolClass | None:
    if not student.school_id:
        return None
    grade, section, curriculum = _class_fields(student)
    raw = student.teaching_classes
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        sc_id = raw[0].get("school_class_id")
        if sc_id is not None and str(sc_id).strip().lstrip("-").isdigit():
            found = db.scalar(
                select(SchoolClass).where(
                    SchoolClass.school_id == student.school_id,
                    SchoolClass.seq == int(sc_id),
                )
            )
            if found:
                return found
    if not grade or not section:
        return None
    stmt = select(SchoolClass).where(
        SchoolClass.school_id == student.school_id,
        SchoolClass.grade == grade,
        func.upper(SchoolClass.section) == section.upper(),
    )
    if curriculum:
        stmt = stmt.where(func.lower(SchoolClass.curriculum) == curriculum.lower())
    return db.scalar(stmt)


def _subjects_for_students(
    db: Session,
    tutor: User,
    students: list[User],
) -> dict[int, list[str]]:
    """Subjects for the tutor Students tab: only the teacher's tagged subjects.

    When the student's class has subject mappings, intersect with those so we
    don't list a tagged subject the class doesn't offer. If the teacher has no
    teaching_subjects, the column is empty (not the full class list).
    """
    if not students:
        return {}
    class_by_student: dict[int, SchoolClass] = {}
    class_ids: set[int] = set()
    for student in students:
        school_class = _school_class_for_student(db, student)
        if school_class:
            class_by_student[student.id] = school_class
            class_ids.add(school_class.id)

    subjects_by_class: dict[int, list[str]] = {cid: [] for cid in class_ids}
    if class_ids:
        classes = list(
            db.scalars(
                select(SchoolClass)
                .where(SchoolClass.id.in_(class_ids))
                .options(selectinload(SchoolClass.subject_links).selectinload(SchoolClassSubject.subject))
            )
        )
        for school_class in classes:
            names = [
                link.subject.name
                for link in school_class.subject_links
                if link.subject and link.subject.is_active
            ]
            subjects_by_class[school_class.id] = names

    tutor_names = [str(n).strip() for n in (tutor.teaching_subjects or []) if str(n).strip()]
    tutor_keys = {n.lower() for n in tutor_names}

    out: dict[int, list[str]] = {}
    for student in students:
        if not tutor_keys:
            out[student.id] = []
            continue
        school_class = class_by_student.get(student.id)
        class_names = list(subjects_by_class.get(school_class.id, [])) if school_class else []
        if class_names:
            class_keys = {n.strip().lower() for n in class_names}
            # Keep teacher tag order; only subjects offered on this student's class.
            names = [n for n in tutor_names if n.lower() in class_keys]
        else:
            # No class map yet — still only show what the teacher tagged.
            names = list(tutor_names)
        out[student.id] = names
    return out


def _completion_map(db: Session, student_ids: list[int]) -> dict[int, int]:
    if not student_ids:
        return {}
    rows = db.execute(
        select(
            StudentChapterProgress.user_id,
            func.avg(StudentChapterProgress.progress_pct),
        )
        .where(StudentChapterProgress.user_id.in_(student_ids))
        .group_by(StudentChapterProgress.user_id)
    ).all()
    return {int(uid): int(round(float(avg or 0))) for uid, avg in rows}


def _assignment_score_avg(db: Session, student_id: int) -> int:
    rows = db.execute(
        select(AssignmentSubmission.score, AssignmentSubmission.max_score).where(
            AssignmentSubmission.student_id == student_id,
            AssignmentSubmission.status.in_(_DONE_SUBMISSION),
        )
    ).all()
    pcts = [_score_pct(score, max_score) for score, max_score in rows]
    pcts = [p for p in pcts if p is not None]
    if not pcts:
        return 0
    return int(round(sum(pcts) / len(pcts)))


def _completion_for_student(db: Session, student_id: int) -> int:
    chapter_avg = _completion_map(db, [student_id]).get(student_id, 0)
    recent_pct = db.scalar(
        select(StudentChapterProgress.progress_pct)
        .where(StudentChapterProgress.user_id == student_id)
        .order_by(StudentChapterProgress.last_accessed_at.desc())
        .limit(1)
    )
    recent = int(recent_pct or 0)
    quiz_avg = _assignment_score_avg(db, student_id)
    # ponytail: best available signal — chapter progress or assignment scores
    return max(chapter_avg, recent, quiz_avg)


def _to_row(
    db: Session,
    student: User,
    *,
    subjects: list[str],
    last_login: datetime | None,
) -> TutorStudentRow:
    grade, section, curriculum = _class_fields(student)
    completion = _completion_for_student(db, student.id)
    # Learning started only when there is measurable progress (login alone ≠ started).
    has_started = completion > 0
    return TutorStudentRow(
        id=student.id,
        slug=_slugify(student.full_name),
        user_id=_username_for(db, student),
        full_name=student.full_name,
        grade=grade,
        section=section,
        curriculum=curriculum,
        grade_label=_grade_label(grade, section, curriculum),
        subjects=subjects,
        completion=completion,
        risk_level=risk_for_student(completion=completion, has_started=has_started),
        last_active=_format_last_active(last_login),
        last_active_at=last_login,
        is_active=student.is_active,
    )


def _filter_assigned(
    rows: list[TutorStudentRow],
    *,
    q: str | None,
    grade: str | None,
    subject: str | None,
    risk_level: str | None,
) -> list[TutorStudentRow]:
    query = (q or "").strip().lower()
    out: list[TutorStudentRow] = []
    for row in rows:
        if grade and grade.lower() != "all" and (row.grade or "") != grade:
            continue
        if subject and subject.lower() != "all":
            want = subject.strip().lower()
            if not any(s.lower() == want for s in row.subjects):
                continue
        if risk_level and risk_level.lower() != "all" and row.risk_level != risk_level:
            continue
        if query and query not in row.full_name.lower() and query not in row.user_id.lower():
            continue
        out.append(row)
    return out


def list_assigned_students(
    db: Session,
    tutor: User,
    *,
    q: str | None = None,
    grade: str | None = None,
    subject: str | None = None,
    risk_level: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> TutorStudentListResponse:
    _require_school_tutor(tutor)
    limit = min(max(1, limit), MAX_PAGE_LIMIT)
    offset = max(0, offset)

    students = list_tutor_assigned_students(db, tutor)
    last_map = _last_login_map(db, [s.id for s in students])
    subjects_map = _subjects_for_students(db, tutor, students)
    rows = [
        _to_row(
            db,
            s,
            subjects=subjects_map.get(s.id, []),
            last_login=last_map.get(s.id),
        )
        for s in students
    ]
    filtered = _filter_assigned(rows, q=q, grade=grade, subject=subject, risk_level=risk_level)
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return TutorStudentListResponse(
        items=page,
        meta=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            default_limit=DEFAULT_PAGE_LIMIT,
        ),
    )


def assigned_student_options(db: Session, tutor: User) -> TutorStudentOptionsResponse:
    _require_school_tutor(tutor)
    students = list_tutor_assigned_students(db, tutor)
    subjects_map = _subjects_for_students(db, tutor, students)
    grades: list[str] = []
    subjects: list[str] = []
    seen_g: set[str] = set()
    seen_s: set[str] = set()
    for student in students:
        grade, _, _ = _class_fields(student)
        if grade and grade not in seen_g:
            seen_g.add(grade)
            grades.append(grade)
        for name in subjects_map.get(student.id, []):
            key = name.lower()
            if key not in seen_s:
                seen_s.add(key)
                subjects.append(name)
    grades.sort(key=lambda g: (len(g), g))
    subjects.sort(key=str.lower)
    return TutorStudentOptionsResponse(
        grades=grades,
        subjects=subjects,
        risk_levels=list(RISK_LEVELS),
        default_limit=DEFAULT_PAGE_LIMIT,
        page_size_options=list(PAGE_SIZE_OPTIONS),
    )


def _format_when(dt: datetime | None) -> str:
    if not dt:
        return "—"
    local = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return local.astimezone(IST).strftime("%d %b %Y, %I:%M %p")


def _format_due(deadline: datetime) -> str:
    local = deadline if deadline.tzinfo else deadline.replace(tzinfo=UTC)
    return local.astimezone(IST).strftime("%d %b %Y")


def _profile_assignment_status(raw: str) -> str:
    if raw in (SubmissionStatus.GRADED.value, SubmissionStatus.SUBMITTED.value):
        return "Completed"
    if raw in (SubmissionStatus.IN_PROGRESS.value, "overdue"):
        return "In Progress"
    return "Not Started"


def _submission_raw_status(assignment: TeacherAssignment, sub: AssignmentSubmission) -> str:
    st = sub.status
    deadline = assignment.deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    if st in (SubmissionStatus.PENDING.value, SubmissionStatus.IN_PROGRESS.value) and deadline < datetime.now(
        UTC
    ):
        return "overdue"
    return st


def _score_pct(score: float | None, max_score: float | None) -> int | None:
    if score is None or max_score is None or max_score <= 0:
        return None
    return int(round((score / max_score) * 100))


def _current_topic(db: Session, student_id: int, fallback: str | None) -> str | None:
    row = db.scalar(
        select(StudentChapterProgress)
        .where(StudentChapterProgress.user_id == student_id)
        .order_by(StudentChapterProgress.last_accessed_at.desc())
        .limit(1)
    )
    if not row:
        return fallback
    chapter = (row.chapter_name or "").strip()
    if chapter:
        return chapter
    subject = (row.subject_name or "").strip()
    return subject or fallback


def _study_session_title(session: StudentStudySession) -> str:
    chapter = (session.chapter_name or "").strip()
    subject = (session.subject_name or "").strip()
    if chapter and subject:
        return f"{subject} — {chapter}"
    return chapter or subject or "AI Tutor session"


def _format_session_range(session: StudentStudySession) -> str:
    start = session.started_at if session.started_at.tzinfo else session.started_at.replace(tzinfo=UTC)
    end = session.ended_at
    if end is None:
        return _format_when(start)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start_local = start.astimezone(IST)
    end_local = end.astimezone(IST)
    start_str = start_local.strftime("%d %b %Y, %I:%M %p")
    if start_local.date() == end_local.date():
        end_str = end_local.strftime("%I:%M %p")
    else:
        end_str = end_local.strftime("%d %b %Y, %I:%M %p")
    return f"{start_str} – {end_str}"


def _ai_activity(db: Session, student_id: int, *, limit: int = 30) -> list[ActivityItem]:
    rows = list(
        db.scalars(
            select(StudentStudySession)
            .where(
                StudentStudySession.user_id == student_id,
                StudentStudySession.ended_at.isnot(None),
                StudentStudySession.duration_seconds >= 30,
            )
            .order_by(StudentStudySession.started_at.desc())
            .limit(limit)
        )
    )
    return [
        ActivityItem(
            title=_study_session_title(session),
            when=_format_session_range(session),
        )
        for session in rows
    ]


def _student_submissions(db: Session, tutor: User, student_id: int) -> list[tuple[TeacherAssignment, AssignmentSubmission]]:
    rows = list(
        db.scalars(
            select(AssignmentSubmission)
            .join(TeacherAssignment, TeacherAssignment.id == AssignmentSubmission.assignment_id)
            .where(
                AssignmentSubmission.student_id == student_id,
                TeacherAssignment.teacher_id == tutor.id,
                TeacherAssignment.status == AssignmentStatus.ACTIVE.value,
            )
            .options(selectinload(AssignmentSubmission.assignment))
            .order_by(TeacherAssignment.deadline.desc())
        )
    )
    out: list[tuple[TeacherAssignment, AssignmentSubmission]] = []
    for sub in rows:
        if sub.assignment:
            out.append((sub.assignment, sub))
    return out


def _profile_assignments(pairs: list[tuple[TeacherAssignment, AssignmentSubmission]], *, limit: int = 30) -> list[AssignmentItem]:
    items: list[AssignmentItem] = []
    for assignment, sub in pairs[:limit]:
        raw = _submission_raw_status(assignment, sub)
        items.append(
            AssignmentItem(
                title=assignment.title,
                due=_format_due(assignment.deadline),
                status=_profile_assignment_status(raw),
            )
        )
    return items


def _quiz_performance(
    pairs: list[tuple[TeacherAssignment, AssignmentSubmission]],
) -> tuple[int, int, list[QuizItem]]:
    quiz_pairs = [(a, s) for a, s in pairs if a.artifact_type == "quiz"]
    scored: list[tuple[datetime, int, str]] = []
    for assignment, sub in quiz_pairs:
        if sub.status not in (SubmissionStatus.SUBMITTED.value, SubmissionStatus.GRADED.value):
            continue
        pct = _score_pct(sub.score, sub.max_score)
        if pct is None or sub.submitted_at is None:
            continue
        ts = sub.submitted_at if sub.submitted_at.tzinfo else sub.submitted_at.replace(tzinfo=UTC)
        scored.append((ts, pct, assignment.title))

    if not scored:
        return 0, 0, []

    scored.sort(key=lambda row: row[0], reverse=True)
    recent = [QuizItem(name=title, score=pct) for _, pct, title in scored[:5]]
    all_scores = [pct for _, pct, _ in scored]
    average = int(round(sum(all_scores) / len(all_scores)))

    now = datetime.now(UTC)
    this_week = [pct for ts, pct, _ in scored if ts >= now - timedelta(days=7)]
    last_week = [pct for ts, pct, _ in scored if now - timedelta(days=14) <= ts < now - timedelta(days=7)]
    trend = 0
    if this_week and last_week:
        trend = int(round(sum(this_week) / len(this_week) - sum(last_week) / len(last_week)))
    elif this_week and not last_week:
        trend = int(round(sum(this_week) / len(this_week) - average))

    return average, trend, recent


def assert_tutor_assigned_student(db: Session, tutor: User, student_id: int) -> User:
    _require_school_tutor(tutor)
    student = require_user_by_account_id(db, student_id)
    if student.role != Role.STUDENT:
        raise AuthException("Not a student account.", status.HTTP_400_BAD_REQUEST)
    assigned = {s.id for s in list_tutor_assigned_students(db, tutor)}
    if student.id not in assigned:
        raise AuthException("Student is not assigned to you.", status.HTTP_403_FORBIDDEN)
    return student


def get_student_profile(db: Session, tutor: User, student_id: int) -> TutorStudentProfileResponse:
    student = assert_tutor_assigned_student(db, tutor, student_id)
    last_map = _last_login_map(db, [student.id])
    subjects_map = _subjects_for_students(db, tutor, [student])
    subject_fallback = (subjects_map.get(student.id) or [None])[0]
    row = _to_row(
        db,
        student,
        subjects=subjects_map.get(student.id, []),
        last_login=last_map.get(student.id),
    )
    strengths: list = []
    needs_improvement: list = []
    teacher_notes = ""
    notes_updated: str | None = None
    risk = row.risk_level

    submission_pairs = _student_submissions(db, tutor, student.id)
    ai_activity = _ai_activity(db, student.id)
    assignments = _profile_assignments(submission_pairs)
    average_quiz, quiz_trend, recent_quizzes = _quiz_performance(submission_pairs)
    current_topic = _current_topic(db, student.id, subject_fallback)
    has_started = row.completion > 0

    try:
        from app.modules.learning_intelligence import service as lia_svc
        from app.modules.teacher.students.schemas import SkillMetric

        twin = lia_svc.fetch_student_profile(db, student.id)
        summary = lia_svc.fetch_teacher_summary(db, student.id)
        strengths = [
            SkillMetric(name=s.replace("_", " "), score=int(twin.confidence_score * 100))
            for s in twin.strong_concepts[:5]
        ] or [SkillMetric(name=s, score=75) for s in summary.strengths[:3]]
        needs_improvement = [
            SkillMetric(name=w.replace("_", " "), score=int((1 - twin.confidence_score) * 100))
            for w in twin.weak_concepts[:5]
        ] or [SkillMetric(name=w, score=40) for w in summary.weaknesses[:3]]
        teacher_notes = "\n".join(summary.observations[:3])
        if summary.recommendations:
            teacher_notes += "\n\nRecommendations:\n" + "\n".join(f"• {r}" for r in summary.recommendations[:4])
        # Keep Not Started until there is real learning progress.
        if has_started:
            lia_risk = _normalize_risk_label(summary.risk_level or twin.risk_level)
            if lia_risk:
                risk = lia_risk
        if summary.period_end:
            notes_updated = summary.period_end.strftime("%d %b %Y")
        db.commit()
    except Exception:
        pass

    payload = row.model_dump()
    payload["risk_level"] = _normalize_risk_label(risk) or row.risk_level
    return TutorStudentProfileResponse(
        **payload,
        current_topic=current_topic,
        overall_progress=row.completion,
        strengths=strengths,
        needs_improvement=needs_improvement,
        ai_activity=ai_activity,
        average_quiz_score=average_quiz,
        quiz_trend=quiz_trend,
        recent_quizzes=recent_quizzes,
        assignments=assignments,
        teacher_notes=teacher_notes,
        notes_updated=notes_updated,
    )


def export_assigned_students_csv(
    db: Session,
    tutor: User,
    *,
    q: str | None = None,
    grade: str | None = None,
    subject: str | None = None,
    risk_level: str | None = None,
) -> str:
    result = list_assigned_students(
        db,
        tutor,
        q=q,
        grade=grade,
        subject=subject,
        risk_level=risk_level,
        limit=MAX_PAGE_LIMIT,
        offset=0,
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "user_id",
            "full_name",
            "grade",
            "section",
            "curriculum",
            "subjects",
            "completion",
            "risk_level",
            "last_active",
            "status",
        ]
    )
    for s in result.items:
        writer.writerow(
            [
                s.user_id,
                s.full_name,
                s.grade or "",
                s.section or "",
                s.curriculum or "",
                "; ".join(s.subjects),
                s.completion,
                s.risk_level,
                s.last_active or "",
                "active" if s.is_active else "inactive",
            ]
        )
    return buf.getvalue()


def _self_check() -> None:
    assert risk_from_completion(80) == RISK_LOW
    assert risk_from_completion(50) == RISK_MEDIUM
    assert risk_from_completion(10) == RISK_HIGH
    assert risk_for_student(completion=0, has_started=False) == RISK_NOT_STARTED
    assert risk_for_student(completion=0, has_started=True) == RISK_NOT_STARTED
    assert risk_for_student(completion=10, has_started=True) == RISK_HIGH
    assert risk_for_student(completion=80, has_started=True) == RISK_LOW
    assert _normalize_risk_label("high") == RISK_HIGH
    assert _normalize_risk_label("not started") == RISK_NOT_STARTED
    assert _normalize_risk_label(None) is None
    assert _grade_label("6", "A", "CBSE") == "6 · A · CBSE"
    assert _slugify("Rahul Sharma") == "rahul-sharma"
    assert _profile_assignment_status("graded") == "Completed"
    assert _profile_assignment_status("in_progress") == "In Progress"
    assert _profile_assignment_status("overdue") == "In Progress"
    assert _profile_assignment_status("pending") == "Not Started"
    assert _score_pct(7, 10) == 70
    assert _score_pct(None, 10) is None
    print("teacher.students.service self-check ok")


if __name__ == "__main__":
    _self_check()

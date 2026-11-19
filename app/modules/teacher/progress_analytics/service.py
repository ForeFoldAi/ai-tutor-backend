"""Build tutor progress analytics from chapter progress, assignments, and LIA."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.auth.service import list_tutor_assigned_students
from app.modules.learning_intelligence import service as lia_svc
from app.modules.school_admin.students.service import _class_fields
from app.modules.student_learning.models import StudentChapterProgress
from app.modules.teacher.assignments.models import AssignmentSubmission, TeacherAssignment
from app.modules.teacher.dashboard.service import _active_this_week, _average_completion
from app.modules.teacher.progress_analytics.schemas import (
    AtRiskStudentOut,
    ClassHealthOut,
    CompletionTrendPointOut,
    ProgressAnalyticsResponse,
    TopicMasteryOut,
)
from app.modules.teacher.students.service import (
    _last_login_map,
    _slugify,
    risk_for_student,
)
from app.modules.users.models import User

_DONE_SUBMISSION = frozenset({"submitted", "graded"})


def _health_status(score: int) -> str:
    if score >= 70:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Needs Attention"


def _assignment_completion_rate(db: Session, tutor_id: int, student_ids: list[int]) -> int:
    if not student_ids:
        return 0
    statuses = db.scalars(
        select(AssignmentSubmission.status)
        .join(TeacherAssignment, TeacherAssignment.id == AssignmentSubmission.assignment_id)
        .where(
            TeacherAssignment.teacher_id == tutor_id,
            AssignmentSubmission.student_id.in_(student_ids),
        )
    ).all()
    if not statuses:
        return 0
    done = sum(1 for status in statuses if status in _DONE_SUBMISSION)
    return int(round(done / len(statuses) * 100))


def _student_completion_map(db: Session, student_ids: list[int]) -> dict[int, int]:
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


def _topic_mastery(db: Session, student_ids: list[int]) -> list[TopicMasteryOut]:
    if not student_ids:
        return []
    topic_expr = func.coalesce(
        func.nullif(func.trim(StudentChapterProgress.chapter_name), ""),
        StudentChapterProgress.subject_name,
    )
    rows = db.execute(
        select(
            topic_expr.label("topic"),
            func.avg(StudentChapterProgress.progress_pct),
        )
        .where(StudentChapterProgress.user_id.in_(student_ids))
        .group_by(topic_expr)
        .order_by(func.avg(StudentChapterProgress.progress_pct).desc())
    ).all()
    out: list[TopicMasteryOut] = []
    for topic, avg in rows:
        label = str(topic or "").strip() or "General"
        out.append(TopicMasteryOut(topic=label, mastery=int(round(float(avg or 0)))))
    return out


def _avg_progress_between(
    db: Session,
    student_ids: list[int],
    start: datetime,
    end: datetime,
) -> int | None:
    if not student_ids:
        return None
    avg = db.scalar(
        select(func.avg(StudentChapterProgress.progress_pct)).where(
            StudentChapterProgress.user_id.in_(student_ids),
            StudentChapterProgress.last_accessed_at >= start,
            StudentChapterProgress.last_accessed_at < end,
        )
    )
    if avg is None:
        return None
    return int(round(float(avg)))


def _completion_trend(db: Session, student_ids: list[int]) -> list[CompletionTrendPointOut]:
    now = datetime.now(UTC)
    points: list[CompletionTrendPointOut] = []
    last_value = 0
    for offset in range(4, -1, -1):
        week_end = now - timedelta(days=offset * 7)
        week_start = week_end - timedelta(days=7)
        value = _avg_progress_between(db, student_ids, week_start, week_end)
        if value is None:
            value = last_value
        else:
            last_value = value
        points.append(
            CompletionTrendPointOut(
                date=week_end.strftime("%b %d"),
                completion=value,
            )
        )
    return points


def _health_trend(db: Session, student_ids: list[int]) -> tuple[int, bool]:
    now = datetime.now(UTC)
    this_avg = _avg_progress_between(db, student_ids, now - timedelta(days=7), now)
    prev_avg = _avg_progress_between(
        db, student_ids, now - timedelta(days=14), now - timedelta(days=7)
    )
    if this_avg is None and prev_avg is None:
        return 0, True
    if this_avg is None:
        this_avg = prev_avg or 0
    if prev_avg is None:
        return 0, True
    delta = this_avg - prev_avg
    return abs(delta), delta >= 0


def _weakest_subject(db: Session, student_id: int, fallback: str) -> str:
    row = db.execute(
        select(StudentChapterProgress.subject_name, StudentChapterProgress.progress_pct)
        .where(StudentChapterProgress.user_id == student_id)
        .order_by(StudentChapterProgress.progress_pct.asc())
        .limit(1)
    ).first()
    if row and row[0]:
        return str(row[0])
    return fallback or "General"


def _at_risk_students(
    db: Session,
    tutor: User,
    students: list[User],
    student_ids: list[int],
    completion_map: dict[int, int],
) -> list[AtRiskStudentOut]:
    seen: set[int] = set()
    out: list[AtRiskStudentOut] = []

    try:
        insights = lia_svc.fetch_class_insights(db, tutor)
        for row in insights.affected_students:
            if row.student_user_id in seen:
                continue
            seen.add(row.student_user_id)
            student = next((s for s in students if s.id == row.student_user_id), None)
            if not student:
                continue
            grade, _, _ = _class_fields(student)
            out.append(
                AtRiskStudentOut(
                    id=student.id,
                    slug=_slugify(student.full_name),
                    name=student.full_name,
                    grade=grade or row.grade or "—",
                    subject=row.topic or "General",
                    risk_level=row.risk_level,
                )
            )
    except Exception:
        pass

    last_login = _last_login_map(db, student_ids)
    stale_cutoff = datetime.now(UTC) - timedelta(days=14)

    for student in students:
        if student.id in seen:
            continue
        completion = completion_map.get(student.id, 0)
        risk = risk_for_student(completion=completion, has_started=completion > 0)
        # At-risk list is for Low/Medium/High only — skip Not Started.
        if risk == "Not Started":
            continue
        if risk == "Low":
            ts = last_login.get(student.id)
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= stale_cutoff:
                    continue
            risk = "Medium"

        grade, _, _ = _class_fields(student)
        subject = _weakest_subject(db, student.id, "General")
        out.append(
            AtRiskStudentOut(
                id=student.id,
                slug=_slugify(student.full_name),
                name=student.full_name,
                grade=grade or "—",
                subject=subject,
                risk_level=risk,
            )
        )

    risk_rank = {"High": 0, "Medium": 1, "Low": 2}
    out.sort(key=lambda row: (risk_rank.get(row.risk_level, 9), row.name.lower()))
    return out[:20]


def build_progress_analytics(db: Session, tutor: User) -> ProgressAnalyticsResponse:
    students = list_tutor_assigned_students(db, tutor)
    student_ids = [s.id for s in students]
    completion_map = _student_completion_map(db, student_ids)

    avg_completion = _average_completion(db, student_ids)
    assignment_rate = _assignment_completion_rate(db, tutor.id, student_ids)
    active_ratio = int(round(_active_this_week(db, student_ids) / len(student_ids) * 100)) if student_ids else 0
    score = min(100, max(0, int(round(0.6 * avg_completion + 0.25 * assignment_rate + 0.15 * active_ratio))))

    trend_percent, trend_up = _health_trend(db, student_ids)

    return ProgressAnalyticsResponse(
        class_health=ClassHealthOut(
            score=score,
            status=_health_status(score),
            trend_percent=trend_percent,
            trend_up=trend_up,
        ),
        topic_mastery=_topic_mastery(db, student_ids),
        completion_trend=_completion_trend(db, student_ids),
        at_risk_students=_at_risk_students(db, tutor, students, student_ids, completion_map),
    )

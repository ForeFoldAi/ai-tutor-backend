"""Build tutor home dashboard — metrics, today's sessions, LIA attention + AI recommendations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.auth.service import list_tutor_assigned_students
from app.modules.learning_intelligence import service as lia_svc
from app.modules.live_sessions import service as live_svc
from app.modules.student_learning.models import StudentChapterProgress
from app.modules.teacher.dashboard.schemas import (
    TutorAiRecommendation,
    TutorAttentionStudent,
    TutorDashboardMetrics,
    TutorDashboardSession,
    TutorDashboardSummaryResponse,
)
from app.modules.teacher.students.service import _last_login_map
from app.modules.users.models import User

_ICON_ACTION: dict[str, tuple[str, str]] = {
    "quiz": ("Create Practice Quiz", "/tutor/lesson-planner"),
    "worksheet": ("Generate Worksheet", "/tutor/lesson-planner"),
    "revision": ("Schedule Revision Session", "/tutor/sessions"),
    "concept": ("Open AI Insights", "/tutor/ai-insights"),
}

_RISK_TAG = {
    "High": "High Risk",
    "Medium": "Needs Support",
    "Low": "Monitor",
}


def _average_completion(db: Session, student_ids: list[int]) -> int:
    if not student_ids:
        return 0
    # Mean of each student's mean chapter progress_pct (0 when no progress rows).
    per_student = {
        int(uid): float(avg or 0)
        for uid, avg in db.execute(
            select(
                StudentChapterProgress.user_id,
                func.avg(StudentChapterProgress.progress_pct),
            )
            .where(StudentChapterProgress.user_id.in_(student_ids))
            .group_by(StudentChapterProgress.user_id)
        ).all()
    }
    values = [per_student.get(sid, 0.0) for sid in student_ids]
    return int(round(sum(values) / len(values)))


def _active_this_week(db: Session, student_ids: list[int]) -> int:
    if not student_ids:
        return 0
    week_ago = datetime.now(UTC) - timedelta(days=7)
    last_map = _last_login_map(db, student_ids)
    count = 0
    for sid in student_ids:
        ts = last_map.get(sid)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts >= week_ago:
            count += 1
    return count


def _is_today(dt: datetime) -> bool:
    now = datetime.now(UTC)
    local = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return local.astimezone(UTC).date() == now.date()


def build_tutor_dashboard(db: Session, tutor: User) -> TutorDashboardSummaryResponse:
    students = list_tutor_assigned_students(db, tutor)
    student_ids = [s.id for s in students]
    total_assigned = len(students)
    active_week = _active_this_week(db, student_ids)
    avg_completion = _average_completion(db, student_ids)

    insights = lia_svc.fetch_class_insights(db, tutor)
    attention = [
        TutorAttentionStudent(
            student_user_id=row.student_user_id,
            name=row.name,
            grade=row.grade,
            topic=row.topic,
            risk_level=row.risk_level,  # type: ignore[arg-type]
            tag=_RISK_TAG.get(row.risk_level, "Needs attention"),
        )
        for row in insights.affected_students[:5]
    ]

    recommendations: list[TutorAiRecommendation] = []
    # Prefer class-level interventions; fall back to weak topics as recommendations.
    for item in insights.interventions[:6]:
        action_label, action_href = _ICON_ACTION.get(
            item.icon, _ICON_ACTION["concept"]
        )
        recommendations.append(
            TutorAiRecommendation(
                id=item.id,
                title=item.title,
                description=item.description,
                priority=item.priority,  # type: ignore[arg-type]
                icon=item.icon,  # type: ignore[arg-type]
                action_label=action_label,
                action_href=action_href,
            )
        )
    if not recommendations:
        for idx, weak in enumerate(insights.weak_topics[:3]):
            recommendations.append(
                TutorAiRecommendation(
                    id=f"weak-{idx}",
                    title=f"{weak.topic} needs focus",
                    description=(
                        f"{weak.student_count} student"
                        f"{'' if weak.student_count == 1 else 's'} struggling "
                        f"({weak.struggle_percent}% struggle signal)"
                    ),
                    priority="High" if weak.struggle_percent >= 60 else "Medium",
                    icon="worksheet",
                    action_label="Generate Worksheet",
                    action_href="/tutor/lesson-planner",
                )
            )

    sessions_out: list[TutorDashboardSession] = []
    try:
        for s in live_svc.list_tutor_sessions(db, tutor):
            if not _is_today(s.starts_at):
                continue
            sessions_out.append(
                TutorDashboardSession(
                    id=s.id,
                    title=s.title,
                    grade=s.grade,
                    section=s.section,
                    subject=s.subject or "",
                    starts_at=s.starts_at,
                    duration_minutes=s.duration_minutes,
                    status=s.status,  # type: ignore[arg-type]
                )
            )
        sessions_out.sort(key=lambda row: row.starts_at)
    except Exception:
        sessions_out = []

    return TutorDashboardSummaryResponse(
        metrics=TutorDashboardMetrics(
            total_assigned=total_assigned,
            active_this_week=active_week,
            average_completion=avg_completion,
            attention_required=len(insights.affected_students),
            assigned_trend=f"{total_assigned} assigned",
            active_trend=f"{active_week} active this week",
            completion_trend=f"{avg_completion}% average",
            attention_trend=f"{len(insights.affected_students)} need attention",
        ),
        todays_sessions=sessions_out[:8],
        attention_students=attention,
        ai_recommendations=recommendations[:6],
    )

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.school_admin.classes.models import SchoolClass
from app.modules.school_admin.classes.validators import get_actor_school, school_curricula
from app.modules.school_admin.dashboard.schemas import (
    DashboardMetrics,
    DashboardSummaryResponse,
    OnboardingProgressItem,
    PendingItem,
)
from app.modules.school_admin.students.service import _class_fields, _tutors_for_student
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User


def _pct(part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return int(round((part / whole) * 100))


def _pct_label(part: int, whole: int) -> str:
    return f"{_pct(part, whole)}% of students"


def build_dashboard_summary(db: Session, actor: User) -> DashboardSummaryResponse:
    if actor.role != Role.SCHOOL_ADMIN:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    school = get_actor_school(db, actor)
    school_id = school.id

    students = list(
        db.scalars(
            select(User).where(User.role == Role.STUDENT, User.school_id == school_id)
        )
    )
    teachers = list(
        db.scalars(
            select(User).where(User.role == Role.TUTOR, User.school_id == school_id)
        )
    )

    now = datetime.now(UTC)
    week_ago = now - timedelta(days=7)
    prev_week_ago = now - timedelta(days=14)

    total_students = len(students)
    total_teachers = len(teachers)

    guided = 0
    self_learning = 0
    without_class = 0
    pending_teacher = 0
    for student in students:
        grade, section, _ = _class_fields(student)
        has_class = bool(grade and section)
        if not has_class:
            without_class += 1
        tutors = _tutors_for_student(db, student) if has_class else []
        if tutors:
            guided += 1
        else:
            self_learning += 1
            if has_class:
                pending_teacher += 1

    school_user_ids = [u.id for u in students + teachers]
    active_this_week = 0
    active_prev_week = 0
    if school_user_ids:
        active_this_week = int(
            db.scalar(
                select(func.count(func.distinct(SessionToken.user_id))).where(
                    SessionToken.user_id.in_(school_user_ids),
                    SessionToken.created_at >= week_ago,
                )
            )
            or 0
        )
        active_prev_week = int(
            db.scalar(
                select(func.count(func.distinct(SessionToken.user_id))).where(
                    SessionToken.user_id.in_(school_user_ids),
                    SessionToken.created_at >= prev_week_ago,
                    SessionToken.created_at < week_ago,
                )
            )
            or 0
        )

    students_week = sum(1 for s in students if s.created_at and s.created_at >= week_ago)
    teachers_week = sum(1 for t in teachers if t.created_at and t.created_at >= week_ago)

    if active_prev_week > 0:
        delta = round(((active_this_week - active_prev_week) / active_prev_week) * 100)
        active_trend = f"{abs(delta)}% {'up' if delta >= 0 else 'down'} from last week"
    else:
        active_trend = f"{active_this_week} this week"

    credentials_users = students + teachers
    creds_generated = sum(1 for u in credentials_users if u.credentials_generated_at)
    creds_not_shared = sum(
        1
        for u in credentials_users
        if u.credentials_generated_at and not u.credentials_shared_at
    )
    creds_total = len(credentials_users)

    metrics = DashboardMetrics(
        total_students=total_students,
        total_teachers=total_teachers,
        teacher_guided_students=guided,
        self_learning_students=self_learning,
        active_this_week=active_this_week,
        students_trend=f"{students_week} this week",
        teachers_trend=f"{teachers_week} this week",
        active_trend=active_trend,
        teacher_guided_percent=_pct_label(guided, total_students),
        self_learning_percent=_pct_label(self_learning, total_students),
    )

    onboarding = [
        OnboardingProgressItem(
            id="teachers_uploaded",
            label="Teachers Uploaded",
            completed=total_teachers,
            total=max(total_teachers, 1) if total_teachers else 0,
            percent=100 if total_teachers else 0,
        ),
        OnboardingProgressItem(
            id="students_uploaded",
            label="Students Uploaded",
            completed=total_students,
            total=max(total_students, 1) if total_students else 0,
            percent=100 if total_students else 0,
        ),
        OnboardingProgressItem(
            id="credentials_generated",
            label="Credentials Generated",
            completed=creds_generated,
            total=creds_total,
            percent=_pct(creds_generated, creds_total),
        ),
        OnboardingProgressItem(
            id="students_assigned",
            label="Students Assigned",
            completed=guided,
            total=total_students,
            percent=_pct(guided, total_students),
        ),
    ]
    # Fix zero-total edge: UI expects total >= completed for bars
    for item in onboarding:
        if item.total == 0:
            item.total = 0
            item.completed = 0
            item.percent = 0
        elif item.completed > item.total:
            item.total = item.completed

    pending = [
        PendingItem(
            id="pending_assignments",
            label="Pending Teacher Assignments",
            count=pending_teacher,
            severity="warning",
            icon="assignments",
        ),
        PendingItem(
            id="creds_not_shared",
            label="Credentials Not Shared",
            count=creds_not_shared,
            severity="warning",
            icon="credentials",
        ),
        PendingItem(
            id="without_class",
            label="Students Without Class Mapping",
            count=without_class,
            severity="danger",
            icon="mapping",
        ),
    ]

    curricula = school_curricula(school)
    if not curricula:
        rows = list(
            db.scalars(
                select(SchoolClass.curriculum)
                .where(SchoolClass.school_id == school_id)
                .distinct()
                .order_by(SchoolClass.curriculum)
            )
        )
        curricula = [str(c).strip() for c in rows if str(c).strip()]

    return DashboardSummaryResponse(
        metrics=metrics,
        onboarding=onboarding,
        pending=pending,
        curricula=curricula,
    )

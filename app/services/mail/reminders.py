"""Daily assignment + today's-session reminder fan-out (called by Celery Beat)."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.modules.auth.constants import Role
from app.modules.live_sessions.models import LiveTutorSession
from app.modules.live_sessions.service import _student_class_keys
from app.modules.teacher.assignments.constants import AssignmentStatus, SubmissionStatus
from app.modules.teacher.assignments.models import AssignmentSubmission, TeacherAssignment
from app.modules.users.models import User
from app.services.mail import send_assignment_reminder, send_session_reminder

logger = logging.getLogger(__name__)


def _tz() -> ZoneInfo:
    return ZoneInfo(get_settings().mail_reminder_tz)


def _today_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    """[start, end) of local calendar day in UTC-aware datetimes."""
    tz = _tz()
    local_now = (now or datetime.now(UTC)).astimezone(tz)
    start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def _fmt(dt: datetime) -> str:
    tz = _tz()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(tz).strftime("%d %b %Y, %I:%M %p")


def _gap_sleep(*, first: bool) -> None:
    """Wait between sends so we never blast the SMTP server in one burst."""
    if first:
        return
    gap = get_settings().mail_reminder_send_gap_seconds
    if gap > 0:
        time.sleep(gap)


def _session_students(db: Session, session: LiveTutorSession) -> list[User]:
    grade = (session.grade or "").strip()
    section = (session.section or "").strip().upper()
    students = list(
        db.scalars(
            select(User).where(
                User.role == Role.STUDENT,
                User.is_active.is_(True),
            )
        ).all()
    )
    out: list[User] = []
    for student in students:
        if session.school_id is not None:
            if student.school_id != session.school_id:
                continue
        else:
            if student.created_by != session.tutor_id:
                continue
        keys = _student_class_keys(student)
        if keys and (grade, section) not in keys:
            continue
        if not keys and session.school_id is not None:
            continue
        out.append(student)
    return out


def run_session_reminders(db: Session) -> dict[str, int]:
    """Email today's live-session reminders (one user at a time, with gap)."""
    settings = get_settings()
    if not settings.mail_reminders_enabled:
        logger.info("Mail reminders disabled (MAIL_REMINDERS_ENABLED=false)")
        return {"sessions": 0, "skipped": 1}

    day_start, day_end = _today_window()
    sent = 0
    first = True

    sessions = list(
        db.scalars(
            select(LiveTutorSession).where(
                LiveTutorSession.starts_at >= day_start,
                LiveTutorSession.starts_at < day_end,
            )
        ).all()
    )
    for session in sessions:
        for student in _session_students(db, session):
            if not student.email:
                continue
            _gap_sleep(first=first)
            first = False
            ok = send_session_reminder(
                db,
                user_id=student.id,
                to=student.email,
                student_name=student.full_name,
                title=session.title,
                subject=session.subject,
                starts_at=_fmt(session.starts_at),
                duration_minutes=session.duration_minutes,
                meeting_link=session.meeting_link,
            )
            if ok:
                sent += 1

    logger.info("Session reminders done sent=%s window=%s..%s", sent, day_start.isoformat(), day_end.isoformat())
    return {"sessions": sent, "skipped": 0}


def run_assignment_reminders(db: Session) -> dict[str, int]:
    """Email due-today assignment reminders (one user at a time, with gap)."""
    settings = get_settings()
    if not settings.mail_reminders_enabled:
        logger.info("Mail reminders disabled (MAIL_REMINDERS_ENABLED=false)")
        return {"assignments": 0, "skipped": 1}

    day_start, day_end = _today_window()
    sent = 0
    first = True

    subs = list(
        db.scalars(
            select(AssignmentSubmission)
            .join(TeacherAssignment, TeacherAssignment.id == AssignmentSubmission.assignment_id)
            .where(
                TeacherAssignment.status == AssignmentStatus.ACTIVE.value,
                AssignmentSubmission.status.in_(
                    [SubmissionStatus.PENDING.value, SubmissionStatus.IN_PROGRESS.value]
                ),
                TeacherAssignment.deadline >= day_start,
                TeacherAssignment.deadline < day_end,
            )
            .options(selectinload(AssignmentSubmission.assignment))
        ).all()
    )
    teacher_ids = {s.assignment.teacher_id for s in subs if s.assignment}
    teachers = {
        u.id: u for u in db.scalars(select(User).where(User.id.in_(list(teacher_ids) or [0]))).all()
    }
    students = {
        u.id: u
        for u in db.scalars(select(User).where(User.id.in_([s.student_id for s in subs] or [0]))).all()
    }

    for sub in subs:
        assignment = sub.assignment
        student = students.get(sub.student_id)
        if not assignment or not student or not student.email:
            continue
        teacher = teachers.get(assignment.teacher_id)
        _gap_sleep(first=first)
        first = False
        ok = send_assignment_reminder(
            db,
            user_id=student.id,
            to=student.email,
            student_name=student.full_name,
            title=assignment.title,
            deadline=_fmt(assignment.deadline),
            teacher_name=teacher.full_name if teacher else None,
            subject=assignment.subject,
        )
        if ok:
            sent += 1

    logger.info("Assignment reminders done sent=%s window=%s..%s", sent, day_start.isoformat(), day_end.isoformat())
    return {"assignments": sent, "skipped": 0}

"""Outbound email: credentials, password OTP, assignment/session reminders."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.modules.users.models import UserSettings
from app.services.mail.messages import (
    build_assignment_reminder_message,
    build_credentials_message,
    build_password_otp_message,
    build_session_reminder_message,
)
from app.services.mail.smtp import send_message

logger = logging.getLogger(__name__)


def send_credentials_email(
    *,
    to: str,
    full_name: str,
    username: str,
    password: str,
    role_label: str,
) -> bool:
    msg = build_credentials_message(
        full_name=full_name,
        username=username,
        password=password,
        role_label=role_label,
    )
    return send_message(
        to=to,
        subject=msg.subject,
        text_body=msg.text_body,
        html_body=msg.html_body,
    )


def send_password_otp_email(
    *,
    to: str,
    full_name: str,
    otp: str,
    expire_minutes: int | None = None,
) -> bool:
    settings = get_settings()
    minutes = expire_minutes if expire_minutes is not None else settings.password_otp_expire_minutes
    msg = build_password_otp_message(
        full_name=full_name,
        otp=otp,
        expire_minutes=minutes,
    )
    return send_message(
        to=to,
        subject=msg.subject,
        text_body=msg.text_body,
        html_body=msg.html_body,
    )


def _notify_ok(db: Session, user_id: int, *, assignment: bool = False, session: bool = False) -> bool:
    row = db.get(UserSettings, user_id)
    if row is None:
        return True
    if not row.notify_email:
        return False
    if assignment and not row.notify_assignments:
        return False
    if session and not row.notify_sessions:
        return False
    return True


def send_assignment_reminder(
    db: Session,
    *,
    user_id: int,
    to: str,
    student_name: str,
    title: str,
    deadline: str,
    teacher_name: str | None = None,
    subject: str | None = None,
) -> bool:
    """Send assignment reminder if user opted in. Returns True if sent/skipped OK, False on SMTP failure."""
    if not _notify_ok(db, user_id, assignment=True):
        logger.info("Skip assignment reminder user_id=%s (notify prefs)", user_id)
        return True
    msg = build_assignment_reminder_message(
        student_name=student_name,
        title=title,
        deadline=deadline,
        teacher_name=teacher_name,
        subject=subject,
    )
    return send_message(
        to=to,
        subject=msg.subject,
        text_body=msg.text_body,
        html_body=msg.html_body,
    )


def send_session_reminder(
    db: Session,
    *,
    user_id: int,
    to: str,
    student_name: str,
    title: str,
    subject: str,
    starts_at: str,
    duration_minutes: int,
    meeting_link: str | None = None,
) -> bool:
    """Send today's session reminder if user opted in."""
    if not _notify_ok(db, user_id, session=True):
        logger.info("Skip session reminder user_id=%s (notify prefs)", user_id)
        return True
    msg = build_session_reminder_message(
        student_name=student_name,
        title=title,
        subject=subject,
        starts_at=starts_at,
        duration_minutes=duration_minutes,
        meeting_link=meeting_link,
    )
    return send_message(
        to=to,
        subject=msg.subject,
        text_body=msg.text_body,
        html_body=msg.html_body,
    )

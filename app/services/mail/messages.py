"""Build subject + multipart bodies from named templates."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.services.mail.render import render_template


@dataclass(frozen=True)
class BuiltMessage:
    subject: str
    text_body: str
    html_body: str


def _with_app(context: dict) -> dict:
    settings = get_settings()
    out = {"app_name": settings.app_name, **context}
    return out


def build_credentials_message(
    *,
    full_name: str,
    username: str,
    password: str,
    role_label: str,
) -> BuiltMessage:
    ctx = _with_app(
        {
            "full_name": full_name,
            "username": username,
            "password": password,
            "role_label": role_label,
        }
    )
    return BuiltMessage(
        subject=f"Your {ctx['app_name']} login credentials ({username})",
        text_body=render_template("credentials.txt", ctx),
        html_body=render_template("credentials.html", ctx),
    )


def build_password_otp_message(*, full_name: str, otp: str, expire_minutes: int) -> BuiltMessage:
    ctx = _with_app(
        {
            "full_name": full_name,
            "otp": otp,
            "expire_minutes": expire_minutes,
        }
    )
    return BuiltMessage(
        subject=f"Your {ctx['app_name']} password reset code",
        text_body=render_template("password_otp.txt", ctx),
        html_body=render_template("password_otp.html", ctx),
    )


def build_signup_pending_admin_message(
    *,
    full_name: str,
    username: str,
    email: str,
    phone: str | None,
    role_label: str,
    school_name: str | None = None,
) -> BuiltMessage:
    ctx = _with_app(
        {
            "full_name": full_name,
            "username": username,
            "email": email,
            "phone": phone or "",
            "role_label": role_label,
            "school_name": school_name or "",
        }
    )
    return BuiltMessage(
        subject=f"New {role_label} signup awaiting approval: {full_name}",
        text_body=render_template("signup_pending_admin.txt", ctx),
        html_body=render_template("signup_pending_admin.html", ctx),
    )


def build_account_approved_message(*, full_name: str) -> BuiltMessage:
    ctx = _with_app({"full_name": full_name})
    return BuiltMessage(
        subject=f"Your {ctx['app_name']} account is approved",
        text_body=render_template("account_approved.txt", ctx),
        html_body=render_template("account_approved.html", ctx),
    )


def build_assignment_reminder_message(
    *,
    student_name: str,
    title: str,
    deadline: str,
    teacher_name: str | None = None,
    subject: str | None = None,
) -> BuiltMessage:
    ctx = _with_app(
        {
            "student_name": student_name,
            "title": title,
            "deadline": deadline,
            "teacher_name": teacher_name or "",
            "subject": subject or "",
        }
    )
    return BuiltMessage(
        subject=f"Reminder: assignment due — {title}",
        text_body=render_template("assignment_reminder.txt", ctx),
        html_body=render_template("assignment_reminder.html", ctx),
    )


def build_session_reminder_message(
    *,
    student_name: str,
    title: str,
    subject: str,
    starts_at: str,
    duration_minutes: int,
    meeting_link: str | None = None,
) -> BuiltMessage:
    ctx = _with_app(
        {
            "student_name": student_name,
            "title": title,
            "subject": subject,
            "starts_at": starts_at,
            "duration_minutes": duration_minutes,
            "meeting_link": meeting_link or "",
        }
    )
    return BuiltMessage(
        subject=f"Today's session: {title}",
        text_body=render_template("session_reminder.txt", ctx),
        html_body=render_template("session_reminder.html", ctx),
    )

"""Render-only self-check for mail templates (no SMTP)."""

from __future__ import annotations

from app.services.mail.messages import (
    build_account_approved_message,
    build_assignment_reminder_message,
    build_credentials_message,
    build_password_otp_message,
    build_session_reminder_message,
    build_signup_pending_admin_message,
)


def _assert_msg(label: str, subject: str, text: str, html: str) -> None:
    assert subject.strip(), f"{label}: empty subject"
    assert text.strip(), f"{label}: empty text"
    assert html.strip(), f"{label}: empty html"
    assert "{{" not in text and "{{" not in html, f"{label}: unrendered placeholders"


def main() -> None:
    cred = build_credentials_message(
        full_name="Anita Verma",
        username="anita.verma",
        password="Av@averma",
        role_label="Student",
    )
    _assert_msg("credentials", cred.subject, cred.text_body, cred.html_body)
    assert "anita.verma" in cred.text_body
    assert "Av@averma" in cred.html_body
    assert "anita.verma" in cred.subject
    from app.core.config import get_settings

    assert get_settings().app_name == "AI Tutor"
    assert "Backend" not in cred.subject
    assert "Backend" not in cred.html_body

    otp = build_password_otp_message(full_name="Anita Verma", otp="482913", expire_minutes=10)
    _assert_msg("password_otp", otp.subject, otp.text_body, otp.html_body)
    assert "482913" in otp.text_body

    pend = build_signup_pending_admin_message(
        full_name="Anita Verma",
        username="anita.verma",
        email="anita@example.com",
        phone=None,
        role_label="School Admin",
        school_name="Sunrise Public School",
    )
    _assert_msg("signup_pending_admin", pend.subject, pend.text_body, pend.html_body)
    assert "anita@example.com" in pend.text_body
    assert "Sunrise Public School" in pend.html_body

    appr = build_account_approved_message(full_name="Anita Verma")
    _assert_msg("account_approved", appr.subject, appr.text_body, appr.html_body)
    assert "approved" in appr.subject

    asgn = build_assignment_reminder_message(
        student_name="Anita Verma",
        title="Fractions worksheet",
        deadline="2026-07-22 18:00 IST",
        teacher_name="Ms. Rao",
        subject="Math",
    )
    _assert_msg("assignment_reminder", asgn.subject, asgn.text_body, asgn.html_body)
    assert "Fractions worksheet" in asgn.html_body

    sess = build_session_reminder_message(
        student_name="Anita Verma",
        title="Live doubt clearing",
        subject="Science",
        starts_at="2026-07-21 16:00 IST",
        duration_minutes=45,
        meeting_link="https://meet.example.com/abc",
    )
    _assert_msg("session_reminder", sess.subject, sess.text_body, sess.html_body)
    assert "Live doubt clearing" in sess.text_body
    print("mail selfcheck ok")


if __name__ == "__main__":
    main()

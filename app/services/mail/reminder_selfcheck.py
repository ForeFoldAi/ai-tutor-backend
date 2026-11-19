"""Self-check for reminder schedule settings + day window (no SMTP / DB)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.config import get_settings
from app.services.mail.reminders import _today_window


def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.mail_session_reminder_hour == 8
    assert settings.mail_assignment_reminder_hour == 9
    assert settings.mail_reminder_send_gap_seconds == 1.0
    start, end = _today_window(datetime(2026, 7, 21, 3, 30, tzinfo=UTC))  # 09:00 IST
    assert start.hour == 18 and start.day == 20
    assert (end - start).total_seconds() == 86400
    print(
        "mail reminders selfcheck ok",
        f"sessions={settings.mail_session_reminder_hour}:00",
        f"assignments={settings.mail_assignment_reminder_hour}:00",
        settings.mail_reminder_tz,
        f"gap={settings.mail_reminder_send_gap_seconds}s",
    )


if __name__ == "__main__":
    main()

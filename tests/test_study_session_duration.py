"""Study session duration: start + end only (no periodic heartbeat)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.main import app  # noqa: F401 — app import order avoids circular imports
from app.modules.student_learning.service import (
    _SESSION_MAX_DURATION,
    _apply_session_duration,
)


def test_apply_session_duration_from_started_at():
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    session = SimpleNamespace(started_at=start, duration_seconds=0, last_heartbeat_at=start)
    now = start + timedelta(minutes=15)
    secs = _apply_session_duration(session, now)
    assert secs == 900
    assert session.duration_seconds == 900


def test_apply_session_duration_caps_overnight_tab():
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    session = SimpleNamespace(started_at=start, duration_seconds=0, last_heartbeat_at=start)
    now = start + timedelta(hours=10)
    secs = _apply_session_duration(session, now)
    assert secs == _SESSION_MAX_DURATION

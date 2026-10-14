"""Student profile data helpers (no DB)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.modules.student_learning.models import StudentStudySession
from app.modules.teacher.students.service import (
    _format_session_range,
    _profile_assignment_status,
    _score_pct,
    _study_session_title,
)


def test_profile_assignment_status_mapping() -> None:
    assert _profile_assignment_status("submitted") == "Completed"
    assert _profile_assignment_status("overdue") == "In Progress"
    assert _profile_assignment_status("pending") == "Not Started"


def test_score_pct() -> None:
    assert _score_pct(4, 5) == 80
    assert _score_pct(0, 0) is None


def test_study_session_title() -> None:
    session = StudentStudySession(
        user_id=1,
        subject_name="Science",
        chapter_name="Chapter 1",
        mode="ai_tutor",
        scope_key="",
    )
    assert _study_session_title(session) == "Science — Chapter 1"


def test_format_session_range_same_day() -> None:
    ist = ZoneInfo("Asia/Kolkata")
    session = StudentStudySession(
        user_id=1,
        subject_name="Science",
        chapter_name="Chapter 1",
        mode="ai_tutor",
        scope_key="",
        started_at=datetime(2026, 7, 22, 9, 0, tzinfo=ist),
        ended_at=datetime(2026, 7, 22, 9, 45, tzinfo=ist),
    )
    assert _format_session_range(session).endswith("– 09:45 AM")
    assert "22 Jul 2026, 09:00 AM" in _format_session_range(session)

"""Self-check: student dashboard response shape (no DB)."""

from __future__ import annotations

from app.modules.catalog.models import BoardEnum, ClassEnum
from app.modules.student_dashboard.schemas import (
    StudentDashboardResponse,
    StudentDashboardStats,
    StudentDashboardSubject,
)


def test_student_dashboard_schema_shape() -> None:
    payload = StudentDashboardResponse(
        overall_progress=42,
        total_chapters=10,
        stats=StudentDashboardStats(
            enrolled_subjects=2,
            lessons_completed=3,
            total_study_seconds=120,
            current_streak=1,
        ),
        subjects=[
            StudentDashboardSubject(
                id=1,
                subject_name="Science",
                board=BoardEnum.CBSE,
                class_level=ClassEnum.CLASS_6,
                progress=42,
                completed_chapters=1,
                total_chapters=5,
                status="in_progress",
            )
        ],
        continue_learning=None,
        recent_lessons=[],
    )
    dumped = payload.model_dump()
    assert dumped["overall_progress"] == 42
    assert dumped["stats"]["enrolled_subjects"] == 2
    assert dumped["subjects"][0]["subject_name"] == "Science"

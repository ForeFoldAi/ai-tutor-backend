"""Build student home-dashboard summary from learning overview."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.modules.student_dashboard.schemas import (
    StudentDashboardContinue,
    StudentDashboardRecentLesson,
    StudentDashboardResponse,
    StudentDashboardStats,
    StudentDashboardSubject,
)
from app.modules.student_learning import service as learning_service
from app.modules.users.models import User

RECENT_LESSONS_LIMIT = 10


def build_student_dashboard(db: Session, user: User) -> StudentDashboardResponse:
    overview = learning_service.get_overview(db, user)

    subjects = [
        StudentDashboardSubject(
            id=s.id,
            subject_name=s.subject_name,
            board=s.board,
            class_level=s.class_level,
            progress=s.progress,
            completed_chapters=s.completed_chapters,
            total_chapters=s.total_chapters,
            status=s.status,
        )
        for s in overview.subjects
    ]

    overall_progress = (
        int(round(sum(s.progress for s in subjects) / len(subjects))) if subjects else 0
    )
    total_chapters = sum(s.total_chapters for s in subjects)

    continue_learning = None
    if overview.continue_learning:
        cl = overview.continue_learning
        continue_learning = StudentDashboardContinue(
            subject_id=cl.subject_id,
            subject_name=cl.subject_name,
            board=cl.board,
            class_level=cl.class_level,
            chapter_id=cl.chapter_id,
            chapter_name=cl.chapter_name,
            file_name=cl.file_name,
            status=cl.status,
            progress=cl.progress,
            last_accessed_at=cl.last_accessed_at,
        )

    # Map chapter progress for recent lessons when available
    chapter_progress: dict[int, int] = {}
    for s in overview.subjects:
        for ch in s.chapters:
            chapter_progress[ch.id] = ch.progress

    recent = [
        StudentDashboardRecentLesson(
            subject_id=r.subject_id,
            subject_name=r.subject_name,
            board=r.board,
            class_level=r.class_level,
            chapter_id=r.chapter_id,
            chapter_name=r.chapter_name,
            file_name=r.file_name,
            status=r.status,
            last_accessed_at=r.last_accessed_at,
            progress=chapter_progress.get(r.chapter_id, 100 if r.status == "completed" else 0),
        )
        for r in overview.recent_lessons[:RECENT_LESSONS_LIMIT]
    ]

    return StudentDashboardResponse(
        overall_progress=overall_progress,
        total_chapters=total_chapters,
        stats=StudentDashboardStats(
            enrolled_subjects=overview.stats.enrolled_subjects,
            lessons_completed=overview.stats.lessons_completed,
            total_study_seconds=overview.stats.total_study_seconds,
            current_streak=overview.stats.current_streak,
        ),
        subjects=subjects,
        continue_learning=continue_learning,
        recent_lessons=recent,
    )

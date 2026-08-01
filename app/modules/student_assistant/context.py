"""Build student-scoped context for the general AI Tutor assistant."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.modules.auth.signup.models import UserSignupProfile
from app.modules.schools.models import School
from app.modules.student_learning.enrollment import (
    resolve_student_school_class,
    student_class_entry,
)
from app.modules.student_learning import service as learning_service
from app.modules.users.models import User

_PLATFORM_GUIDE = """
AI Tutor platform (student-facing):
- AI Learning Studio: home for subjects, quick-start modes, and resume learning.
- Ask AI Tutor (this chat): platform help plus general school questions — including topics outside the textbook.
- Quick Start (Ask / Practice / Explain): textbook-only popup modes that detect subject/topic from your question — no subject/chapter picker.
- Chapter AI Tutor: deeper textbook-grounded chat after picking a subject and chapter from Your Subjects.
- AI Voice: spoken tutoring for a selected chapter.
- My Learning: progress dashboard across subjects.
- Live Classes: join sessions scheduled by your tutor.
- Assignments: work assigned by your tutor or school.
""".strip()


def _enum_list(values) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        out.append(getattr(v, "value", str(v)))
    return out


def build_student_context(db: Session, user: User) -> str:
    """Serialize only this student's data plus their tagged school/tutor."""
    profile = db.get(UserSignupProfile, user.id)
    overview = learning_service.get_overview(db, user)
    school_class = resolve_student_school_class(db, user)
    class_entry = student_class_entry(user)

    school: School | None = None
    if user.school_id is not None:
        school = db.get(School, user.school_id)

    tutor: User | None = None
    if user.created_by is not None:
        tutor = db.get(User, user.created_by)

    lines: list[str] = [
        "STUDENT PROFILE",
        f"- Name: {user.full_name}",
        f"- Email: {user.email}",
    ]

    if profile:
        grades = _enum_list([profile.student_grade] if profile.student_grade else [])
        if grades:
            lines.append(f"- Grade: {grades[0]}")
        curricula = _enum_list(profile.curricula)
        if curricula:
            lines.append(f"- Curriculum/board: {', '.join(curricula)}")
        fav = _enum_list(profile.favorite_subjects)
        if fav:
            lines.append(f"- Favorite subjects: {', '.join(fav)}")
        goals = _enum_list(profile.learning_goals)
        if goals:
            lines.append(f"- Learning goals: {', '.join(goals)}")

    if school:
        branch = f" ({school.branch})" if school.branch else ""
        lines.extend(
            [
                "",
                "SCHOOL",
                f"- Name: {school.name}{branch}",
            ]
        )
        if school.board:
            lines.append(f"- Board: {school.board}")
        school_curricula = _enum_list(school.curricula)
        if school_curricula:
            lines.append(f"- Curricula: {', '.join(school_curricula)}")

    if tutor:
        lines.extend(["", "TAGGED TUTOR", f"- Name: {tutor.full_name}"])

    if school_class:
        lines.extend(
            [
                "",
                "CLASS",
                f"- Grade {school_class.grade} · Section {school_class.section}",
                f"- Curriculum: {school_class.curriculum}",
            ]
        )
    elif class_entry:
        grade = str(class_entry.get("grade", "")).strip()
        sections = class_entry.get("sections") or []
        section = str(sections[0]).strip() if sections else ""
        curriculum = str(class_entry.get("curriculum") or user.teaching_board or "").strip()
        if grade or section:
            lines.extend(["", "CLASS", f"- Grade {grade} · Section {section}".strip(" · ")])
        if curriculum:
            lines.append(f"- Curriculum: {curriculum}")

    stats = overview.stats
    lines.extend(
        [
            "",
            "LEARNING STATS",
            f"- Enrolled subjects: {stats.enrolled_subjects}",
            f"- Lessons completed: {stats.lessons_completed}",
            f"- Total study time (seconds): {stats.total_study_seconds}",
            f"- Current streak (days): {stats.current_streak}",
        ]
    )

    if overview.continue_learning:
        cl = overview.continue_learning
        lines.extend(
            [
                "",
                "CONTINUE LEARNING",
                f"- Subject: {cl.subject_name}",
                f"- Chapter: {cl.chapter_name or cl.file_name}",
                f"- Progress: {cl.progress}%",
            ]
        )

    if overview.subjects:
        lines.extend(["", "ENROLLED SUBJECTS"])
        for subj in overview.subjects:
            lines.append(
                f"- {subj.subject_name} ({subj.board.value}, {subj.class_level.value}): "
                f"{subj.completed_chapters}/{subj.total_chapters} chapters · {subj.progress}%"
            )
            chapter_bits: list[str] = []
            for ch in subj.chapters[:12]:
                label = ch.chapter or ch.file_name
                chapter_bits.append(f"{label} [{ch.status}]")
            if chapter_bits:
                lines.append(f"  Chapters: {'; '.join(chapter_bits)}")
            if len(subj.chapters) > 12:
                lines.append(f"  …and {len(subj.chapters) - 12} more chapters")

    recent = overview.recent_lessons[:5]
    if recent:
        lines.extend(["", "RECENT LESSONS"])
        for lesson in recent:
            lines.append(
                f"- {lesson.subject_name}: {lesson.chapter_name or lesson.file_name} [{lesson.status}]"
            )

    lines.extend(["", "PLATFORM GUIDE", _PLATFORM_GUIDE])
    return "\n".join(lines)


def build_suggested_prompts(db: Session, user: User) -> list[str]:
    overview = learning_service.get_overview(db, user)
    prompts = [
        "What subjects am I enrolled in?",
        "Explain photosynthesis in simple words",
        "How can AI Tutor help me study?",
        "What is AI Learning Studio?",
    ]
    if overview.continue_learning:
        prompts.insert(0, "What was I studying last?")
    if overview.stats.current_streak > 0:
        prompts.append("How is my learning streak going?")
    if user.school_id is not None:
        prompts.append("Tell me about my school and class.")
    elif user.created_by is not None:
        prompts.append("Who is my tutor?")
    return prompts[:6]

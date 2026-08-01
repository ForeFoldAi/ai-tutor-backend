from __future__ import annotations

import json

from app.modules.teacher.lesson_planner.constants import ArtifactType
from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

_SYSTEM = (
    "You are an expert curriculum designer and teacher trainer. "
    "Generate pedagogically sound, age-appropriate classroom materials grounded in the provided textbook context. "
    "Never invent facts that contradict the context. Use clear, actionable language for teachers."
)


def build_artifact_prompt(state: PlannerState, artifact_type: str) -> tuple[str, str]:
    context = state.get("chapter_context") or "No textbook context available."
    objectives = state.get("learning_objectives") or ""
    chapter = state.get("chapter_name") or ""
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    duration = state.get("duration_minutes") or 45
    pedagogy = (state.get("metadata") or {}).get("pedagogy") or {}
    figures = state.get("figures") or []
    experiments = state.get("experiments") or []
    question_bank = (state.get("metadata") or {}).get("question_bank_sample") or []

    label = artifact_type.replace("_", " ").title()
    user = (
        f"Grade/Class: {grade}\n"
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"{topics_focus_block(state)}"
        f"Duration: {duration} minutes\n"
        f"Learning objectives:\n{objectives}\n\n"
        f"Pedagogy analysis:\n{json.dumps(pedagogy, ensure_ascii=False)[:4000]}\n\n"
        f"Textbook context:\n{context[:12000]}\n\n"
    )

    if figures:
        user += f"Available textbook figures:\n{json.dumps(figures[:6], ensure_ascii=False)[:3000]}\n\n"
    if experiments:
        user += f"Interactive experiments:\n{json.dumps(experiments[:2], ensure_ascii=False)[:3000]}\n\n"
    if question_bank and artifact_type in {ArtifactType.WORKSHEET.value, ArtifactType.QUIZ.value}:
        user += f"Question bank samples:\n{json.dumps(question_bank, ensure_ascii=False)}\n\n"

    user += f"Generate: {label}"

    if artifact_type == ArtifactType.LESSON_PLAN.value:
        user += (
            "\nInclude structured activities with timings, prerequisites, key concepts, "
            "assessment plan, revision plan, and suggested visuals from the figure list when relevant."
        )
    elif artifact_type == ArtifactType.WORKSHEET.value:
        user += "\nBalance difficulty across fill-in-the-blanks, T/F, matching, short and long answer."
    elif artifact_type == ArtifactType.QUIZ.value:
        user += "\nInclude MCQ, short answer, HOTS, assertion-reason, and an answer key."

    return _SYSTEM, user

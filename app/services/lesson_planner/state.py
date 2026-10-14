from __future__ import annotations

from typing import Any, TypedDict

from app.modules.teacher.lesson_planner.constants import ArtifactType


class PlannerState(TypedDict, total=False):
    job_id: str
    user_id: str
    lesson_plan_id: str
    grade: str
    subject: str
    board: str | None
    chapter_id: str | None
    chapter_name: str
    duration_minutes: int
    learning_objectives: str
    selected_topics: list[str]
    ppt_template: str
    ppt_slide_count: int
    requested_artifacts: list[str]
    chapter_context: str
    figures: list[dict[str, Any]]
    experiments: list[dict[str, Any]]
    outputs: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    errors: list[str]


def initial_state_from_payload(payload: dict[str, Any], *, job_id: str, user_id: str, lesson_plan_id: str) -> PlannerState:
    artifacts = payload.get("requested_artifacts") or [a.value for a in ArtifactType]
    if artifacts and isinstance(artifacts[0], ArtifactType):
        artifacts = [a.value for a in artifacts]
    return PlannerState(
        job_id=job_id,
        user_id=user_id,
        lesson_plan_id=lesson_plan_id,
        grade=str(payload.get("grade", "")),
        subject=str(payload.get("subject", "")),
        board=payload.get("board"),
        chapter_id=payload.get("chapter_id"),
        chapter_name=str(payload.get("chapter_name", "")),
        duration_minutes=int(payload.get("duration_minutes", 45)),
        learning_objectives=str(payload.get("learning_objectives", "")),
        selected_topics=[
            str(t).strip()
            for t in (payload.get("topics") or payload.get("selected_topics") or [])
            if str(t).strip()
        ],
        ppt_template=str(payload.get("ppt_template") or "clean_academic"),
        ppt_slide_count=int(payload.get("ppt_slide_count") or 12),
        requested_artifacts=list(artifacts),
        chapter_context="",
        figures=[],
        experiments=[],
        outputs={},
        metadata={},
        errors=[],
    )

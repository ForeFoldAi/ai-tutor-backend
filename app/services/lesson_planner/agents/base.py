from __future__ import annotations

from app.modules.teacher.lesson_planner.constants import ArtifactType
from app.services.lesson_planner.fallbacks import fallback_artifact
from app.services.lesson_planner.llm import generate_artifact_markdown, generate_lesson_plan_markdown
from app.services.lesson_planner.prompts import build_artifact_prompt
from app.services.lesson_planner.prompts_lesson_plan import build_lesson_plan_markdown_prompt
from app.services.lesson_planner.prompts_examples import build_examples_markdown_prompt
from app.services.lesson_planner.prompts_teaching_notes import build_teaching_notes_markdown_prompt
from app.services.lesson_planner.prompts_homework import build_homework_markdown_prompt
from app.services.lesson_planner.prompts_ppt_outline import build_ppt_outline_markdown_prompt
from app.services.lesson_planner.prompts_quiz import build_quiz_markdown_prompt
from app.services.lesson_planner.prompts_worksheet import build_worksheet_markdown_prompt
from app.services.lesson_planner.pydantic_ai_agent import generate_structured_artifact
from app.services.lesson_planner.schemas import ARTIFACT_SCHEMAS
from app.services.lesson_planner.state import PlannerState


def _generate_lesson_plan_markdown(state: PlannerState) -> dict:
    system_prompt, user_prompt = build_lesson_plan_markdown_prompt(state)
    md = generate_lesson_plan_markdown(system_prompt=system_prompt, user_prompt=user_prompt)
    if md and len(md.strip()) > 200:
        return {"format": "markdown", "markdown": md}
    return fallback_artifact(state, ArtifactType.LESSON_PLAN)


def _generate_teaching_notes_markdown(state: PlannerState) -> dict:
    system_prompt, user_prompt = build_teaching_notes_markdown_prompt(state)
    md = generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_label="teaching notes",
    )
    if md and len(md.strip()) > 200:
        return {"format": "markdown", "markdown": md}
    return fallback_artifact(state, ArtifactType.TEACHING_NOTES)


def _generate_examples_markdown(state: PlannerState) -> dict:
    system_prompt, user_prompt = build_examples_markdown_prompt(state)
    md = generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_label="examples",
    )
    if md and len(md.strip()) > 200:
        return {"format": "markdown", "markdown": md}
    return fallback_artifact(state, ArtifactType.EXAMPLES)


def _generate_worksheet_markdown(state: PlannerState) -> dict:
    system_prompt, user_prompt = build_worksheet_markdown_prompt(state)
    md = generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_label="worksheet",
    )
    if md and len(md.strip()) > 200:
        return {"format": "markdown", "markdown": md}
    return fallback_artifact(state, ArtifactType.WORKSHEET)


def _generate_quiz_markdown(state: PlannerState) -> dict:
    system_prompt, user_prompt = build_quiz_markdown_prompt(state)
    md = generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_label="quiz",
    )
    if md and len(md.strip()) > 200:
        return {"format": "markdown", "markdown": md}
    return fallback_artifact(state, ArtifactType.QUIZ)


def _generate_homework_markdown(state: PlannerState) -> dict:
    system_prompt, user_prompt = build_homework_markdown_prompt(state)
    md = generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_label="homework",
    )
    if md and len(md.strip()) > 200:
        return {"format": "markdown", "markdown": md}
    return fallback_artifact(state, ArtifactType.HOMEWORK)


def _generate_ppt_outline_markdown(state: PlannerState) -> dict:
    from app.services.lesson_planner.export.deck_schema import ppt_slides_from_markdown

    system_prompt, user_prompt = build_ppt_outline_markdown_prompt(state)
    md = generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=10000,
        log_label="ppt outline",
    )
    if md and len(md.strip()) > 200:
        slides = ppt_slides_from_markdown(
            md,
            chapter=state.get("chapter_name"),
            subject=state.get("subject"),
            max_slides=int(state.get("ppt_slide_count") or 12),
        )
        return {
            "format": "markdown",
            "markdown": md,
            "slides": slides,
            "template_id": state.get("ppt_template") or "clean_academic",
            "slide_count_target": int(state.get("ppt_slide_count") or 12),
            "chapter_name": state.get("chapter_name") or "",
            "subject": state.get("subject") or "",
            "figures": list(state.get("figures") or [])[:8],
        }
    return fallback_artifact(state, ArtifactType.PPT_OUTLINE)


def run_artifact_agent(state: PlannerState, artifact_type: str) -> dict:
    if artifact_type == ArtifactType.LESSON_PLAN.value:
        return _generate_lesson_plan_markdown(state)
    if artifact_type == ArtifactType.TEACHING_NOTES.value:
        return _generate_teaching_notes_markdown(state)
    if artifact_type == ArtifactType.EXAMPLES.value:
        return _generate_examples_markdown(state)
    if artifact_type == ArtifactType.WORKSHEET.value:
        return _generate_worksheet_markdown(state)
    if artifact_type == ArtifactType.QUIZ.value:
        return _generate_quiz_markdown(state)
    if artifact_type == ArtifactType.HOMEWORK.value:
        return _generate_homework_markdown(state)
    if artifact_type == ArtifactType.PPT_OUTLINE.value:
        return _generate_ppt_outline_markdown(state)

    system_prompt, user_prompt = build_artifact_prompt(state, artifact_type)
    schema_cls = ARTIFACT_SCHEMAS.get(artifact_type)
    result = generate_structured_artifact(
        artifact_type=artifact_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_cls=schema_cls,
    )
    if not result:
        return fallback_artifact(state, ArtifactType(artifact_type))

    return result

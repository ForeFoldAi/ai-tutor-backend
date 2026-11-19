"""Lesson planner artifact adapters (schema-level + agent hooks)."""

from __future__ import annotations

from typing import Any


ARTIFACT_KINDS = (
    "lesson_plan",
    "teaching_notes",
    "worksheet",
    "quiz",
    "homework",
    "examples",
    "ppt_outline",
)


def schema_for(kind: str) -> type:
    from app.services.lesson_planner.schemas import ARTIFACT_SCHEMAS

    if kind not in ARTIFACT_SCHEMAS:
        raise KeyError(f"Unknown artifact kind: {kind}")
    return ARTIFACT_SCHEMAS[kind]


def validate_artifact_payload(kind: str, payload: dict[str, Any]) -> tuple[bool, str, Any]:
    """Validate a dict against the production Pydantic schema."""
    model = schema_for(kind)
    try:
        obj = model.model_validate(payload)
        return True, "schema_ok", obj.model_dump()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), None


def generate_artifact_offline(kind: str, topic: str, context: str) -> dict[str, Any]:
    """
    Call the lesson-planner agent for ``kind`` when available.

    Falls back to a deterministic minimal valid payload so schema/quality
    checks still run in offline CI (marked in meta).
    """
    try:
        from app.services.lesson_planner import agents as lp_agents  # type: ignore

        agent_map = {
            "lesson_plan": getattr(lp_agents, "lesson_plan", None),
            "quiz": getattr(lp_agents, "quiz", None),
            "worksheet": getattr(lp_agents, "worksheet", None),
            "homework": getattr(lp_agents, "homework", None),
            "teaching_notes": getattr(lp_agents, "teaching_notes", None),
        }
        mod = agent_map.get(kind)
        if mod is not None and hasattr(mod, "generate"):
            out = mod.generate(topic=topic, context=context)
            if hasattr(out, "model_dump"):
                return {"source": "agent", "payload": out.model_dump()}
            if isinstance(out, dict):
                return {"source": "agent", "payload": out}
    except Exception as exc:  # noqa: BLE001
        return {"source": "fallback", "error": str(exc), "payload": _minimal_payload(kind, topic)}

    return {"source": "fallback", "payload": _minimal_payload(kind, topic)}


def _minimal_payload(kind: str, topic: str) -> dict[str, Any]:
    """Smallest structurally valid payload for offline schema checks."""
    if kind == "lesson_plan":
        return {
            "title": f"Lesson: {topic}",
            "duration": 40,
            "objectives": [f"Understand {topic}"],
            "prerequisites": [],
            "key_concepts": [topic],
            "activities": [
                {
                    "title": "Explain",
                    "duration_minutes": 15,
                    "description": f"Teach {topic}",
                    "activity_type": "instruction",
                }
            ],
            "assessment_plan": "Oral questions",
            "revision_plan": "Summarize key points",
            "visuals": [],
        }
    if kind == "quiz":
        return {
            "mcq": [
                {
                    "question": f"What is {topic}?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                }
            ],
            "short_answer": [],
            "hots_questions": [],
            "assertion_reason": [],
            "answers": {"q1": "A"},
        }
    if kind == "worksheet":
        return {
            "fill_blanks": [
                {"prompt": f"{topic} is ____", "answer": "important"},
                {"prompt": f"One key idea in {topic} is ____", "answer": "concept"},
            ],
            "true_false": [{"statement": f"{topic} is part of the curriculum.", "answer": True}],
            "match_following": [],
            "short_answer": [{"question": f"Define {topic}"}],
            "long_answer": [],
            "application_questions": [{"question": f"Give one real-life example of {topic}."}],
        }
    if kind == "homework":
        return {
            "practice_questions": [
                {"question": f"Practice: summarize {topic} in 5 lines."},
                {"question": f"List three facts about {topic}."},
            ],
            "observation_tasks": [{"task": f"Observe something related to {topic} at home."}],
            "project_work": [],
            "reading_assignment": [{"task": f"Re-read the section on {topic}."}],
            "revision_tasks": [{"task": f"Revise key terms from {topic}."}],
        }
    if kind == "teaching_notes":
        return {
            "introduction_script": f"Today we study {topic}.",
            "teacher_explanation": f"Explain {topic} clearly.",
            "common_mistakes": ["Confusing related terms"],
            "real_life_connections": [],
            "questions_to_ask": [f"What is {topic}?"],
            "blackboard_flow": [topic],
        }
    return {}

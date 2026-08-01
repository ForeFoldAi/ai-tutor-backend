"""Shared prompt snippets for lesson planner."""

from __future__ import annotations

from app.services.lesson_planner.state import PlannerState


def topics_focus_block(state: PlannerState) -> str:
    """Instruct the model to cover only teacher-selected topics when present."""
    topics = [str(t).strip() for t in (state.get("selected_topics") or []) if str(t).strip()]
    if not topics:
        return ""
    joined = "; ".join(topics)
    return (
        "FOCUS TOPICS FOR THIS LESSON (cover ONLY these; do not expand to the rest of the chapter "
        f"unless needed as a brief prerequisite):\n{joined}\n\n"
    )

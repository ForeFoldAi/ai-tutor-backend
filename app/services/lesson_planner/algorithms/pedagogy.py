from __future__ import annotations

from typing import Any

from app.services.lesson_planner.algorithms.bloom import classify_objectives
from app.services.lesson_planner.algorithms.concepts import extract_key_concepts
from app.services.lesson_planner.algorithms.curriculum import align_curriculum
from app.services.lesson_planner.algorithms.difficulty import estimate_difficulty
from app.services.lesson_planner.algorithms.misconceptions import detect_misconceptions
from app.services.lesson_planner.algorithms.prerequisite_dag import build_prerequisite_dag
from app.services.lesson_planner.algorithms.revision_schedule import generate_revision_schedule
from app.services.lesson_planner.state import PlannerState


def enrich_pedagogy_metadata(state: PlannerState) -> dict[str, Any]:
    """Run pedagogy algorithms and return metadata block for prompts + persistence."""
    context = state.get("chapter_context") or ""
    objectives_raw = state.get("learning_objectives") or ""
    objectives = [o.strip() for o in objectives_raw.split("\n") if o.strip()]
    if not objectives and objectives_raw.strip():
        objectives = [objectives_raw.strip()]

    concepts = extract_key_concepts(context)
    bloom = classify_objectives(objectives) if objectives else []
    difficulty = estimate_difficulty(
        f"{state.get('chapter_name', '')} {objectives_raw}",
        grade=state.get("grade", ""),
    )
    prereq = build_prerequisite_dag(concepts)
    revision = generate_revision_schedule(concepts)
    curriculum = align_curriculum(
        grade=state.get("grade", ""),
        subject=state.get("subject", ""),
        chapter_name=state.get("chapter_name", ""),
        concepts=concepts,
    )
    misconceptions = detect_misconceptions(context)

    return {
        "key_concepts": concepts,
        "bloom_mapping": bloom,
        "estimated_difficulty": difficulty,
        "prerequisite_dag": prereq,
        "revision_schedule": revision,
        "curriculum_alignment": curriculum,
        "misconceptions": misconceptions,
    }

from __future__ import annotations

import logging
from typing import Any

from app.services.science_experiment.experiment_catalog import match_science_experiment

logger = logging.getLogger(__name__)

_SCIENCE_KEYWORDS = frozenset(
    {
        "science",
        "physics",
        "chemistry",
        "biology",
        "experiment",
        "lab",
        "motion",
        "force",
        "energy",
        "cell",
        "plant",
        "animal",
        "electricity",
        "magnet",
        "light",
        "sound",
        "water",
        "air",
        "matter",
    }
)


def _is_science_subject(subject: str) -> bool:
    s = (subject or "").lower()
    return any(kw in s for kw in _SCIENCE_KEYWORDS)


def retrieve_experiments_for_lesson(
    *,
    subject: str,
    grade: str,
    chapter_name: str,
    learning_objectives: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Match interactive science experiments from science_experiment catalog."""
    if not _is_science_subject(subject):
        return []

    query = f"{chapter_name} {learning_objectives}".strip()
    try:
        primary = match_science_experiment(query, grade)
        experiments: list[dict[str, Any]] = [primary] if primary else []

        # ponytail: single catalog match today — Phase 3 adds Chroma experiment bank
        if chapter_name and chapter_name.lower() not in (primary or {}).get("conceptName", "").lower():
            alt = match_science_experiment(chapter_name, grade)
            if alt and alt != primary:
                experiments.append(alt)

        return experiments[:limit]
    except Exception as exc:
        logger.warning("Experiment retrieval failed: %s", exc)
        return []

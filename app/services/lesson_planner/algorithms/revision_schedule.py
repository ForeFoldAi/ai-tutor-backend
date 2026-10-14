from __future__ import annotations

from typing import Any


def generate_revision_schedule(concepts: list[str], *, days: int = 5) -> list[dict[str, Any]]:
    """Spaced revision plan across concepts."""
    if not concepts:
        return []
    schedule: list[dict[str, Any]] = []
    for day in range(1, days + 1):
        concept = concepts[(day - 1) % len(concepts)]
        schedule.append(
            {
                "day": day,
                "focus": concept,
                "task": f"Review and recall key points about {concept}",
                "duration_minutes": 15,
            }
        )
    return schedule

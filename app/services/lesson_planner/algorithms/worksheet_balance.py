from __future__ import annotations

from typing import Any

from app.services.lesson_planner.algorithms.difficulty import estimate_difficulty


def balance_worksheet_questions(
    questions: list[dict[str, Any]],
    *,
    grade: str,
    target_easy: int = 2,
    target_medium: int = 2,
    target_hard: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Bucket worksheet items by estimated difficulty."""
    buckets: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for q in questions:
        text = str(q.get("question") or q.get("statement") or q)
        diff = estimate_difficulty(text, grade=grade)
        if diff < 2.5:
            buckets["easy"].append({**q, "difficulty": diff})
        elif diff < 3.8:
            buckets["medium"].append({**q, "difficulty": diff})
        else:
            buckets["hard"].append({**q, "difficulty": diff})

    return {
        "easy": buckets["easy"][:target_easy],
        "medium": buckets["medium"][:target_medium],
        "hard": buckets["hard"][:target_hard],
        "distribution": {k: len(v) for k, v in buckets.items()},
    }

from __future__ import annotations

import re

from app.services.lesson_planner.algorithms.bloom import BloomLevel, classify_bloom

_GRADE_NUM_RE = re.compile(r"class[_\s]*(\d{1,2})", re.I)


def _grade_number(grade: str) -> int:
    m = _GRADE_NUM_RE.search((grade or "").replace("_", " "))
    return int(m.group(1)) if m else 6


def estimate_difficulty(
    text: str,
    *,
    grade: str = "",
    bloom_level: BloomLevel | None = None,
) -> float:
    """Estimate difficulty on 1.0 (easy) – 5.0 (hard) scale."""
    level = bloom_level or classify_bloom(text)
    bloom_scores = {
        BloomLevel.REMEMBER: 1.0,
        BloomLevel.UNDERSTAND: 2.0,
        BloomLevel.APPLY: 3.0,
        BloomLevel.ANALYZE: 3.5,
        BloomLevel.EVALUATE: 4.0,
        BloomLevel.CREATE: 4.5,
    }
    base = bloom_scores.get(level, 2.5)
    grade_factor = min(1.5, _grade_number(grade) / 10.0)
    length_factor = min(1.0, len((text or "").split()) / 200.0)
    return round(min(5.0, max(1.0, base + grade_factor * 0.3 + length_factor * 0.2)), 2)

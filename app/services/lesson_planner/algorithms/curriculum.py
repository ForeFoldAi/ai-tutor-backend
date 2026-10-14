from __future__ import annotations

import re
from typing import Any

_GRADE_RE = re.compile(r"class[_\s]*(\d{1,2})", re.I)


def align_curriculum(
    *,
    grade: str,
    subject: str,
    chapter_name: str,
    concepts: list[str],
) -> dict[str, Any]:
    m = _GRADE_RE.search((grade or "").replace("_", " "))
    grade_num = int(m.group(1)) if m else None
    band = "primary" if grade_num and grade_num <= 5 else "middle" if grade_num and grade_num <= 8 else "secondary"
    return {
        "grade": grade,
        "grade_number": grade_num,
        "subject": subject,
        "chapter": chapter_name,
        "curriculum_band": band,
        "aligned_concepts": concepts[:8],
        "standards_note": f"Aligned to {grade} {subject} — {chapter_name}",
    }

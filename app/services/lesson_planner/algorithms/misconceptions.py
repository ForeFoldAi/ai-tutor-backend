from __future__ import annotations

import re

_MISCONCEPTION_MARKERS = (
    re.compile(r"\b(common mistake|misconception|students often|confuse|incorrectly think)\b", re.I),
    re.compile(r"\b(do not|don't|never)\s+\w+\s+with\b", re.I),
)


def detect_misconceptions(context: str, *, max_items: int = 5) -> list[str]:
    """Extract misconception hints from textbook context."""
    if not context:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", context)
    found: list[str] = []
    for sentence in sentences:
        if any(pat.search(sentence) for pat in _MISCONCEPTION_MARKERS):
            found.append(sentence.strip()[:300])
        if len(found) >= max_items:
            break
    return found

from __future__ import annotations

import re
from enum import StrEnum

_BLOOM_PATTERNS: dict[str, re.Pattern[str]] = {
    "remember": re.compile(r"\b(define|list|name|identify|recall|state|label)\b", re.I),
    "understand": re.compile(r"\b(explain|describe|summarize|interpret|classify)\b", re.I),
    "apply": re.compile(r"\b(solve|calculate|apply|demonstrate|use|compute)\b", re.I),
    "analyze": re.compile(r"\b(compare|contrast|analyze|differentiate|examine)\b", re.I),
    "evaluate": re.compile(r"\b(evaluate|justify|assess|critique|defend)\b", re.I),
    "create": re.compile(r"\b(design|construct|create|develop|formulate|invent)\b", re.I),
}


class BloomLevel(StrEnum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


def classify_bloom(text: str) -> BloomLevel:
    scores = {level: len(pat.findall(text or "")) for level, pat in _BLOOM_PATTERNS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return BloomLevel.UNDERSTAND
    return BloomLevel(best)


def classify_objectives(objectives: list[str]) -> list[dict[str, str]]:
    return [{"objective": obj, "bloom_level": classify_bloom(obj).value} for obj in objectives if obj.strip()]

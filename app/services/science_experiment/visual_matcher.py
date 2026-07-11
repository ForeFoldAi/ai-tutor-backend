"""Priority-based experiment matcher for science queries."""

from __future__ import annotations

import re
from typing import Any, Callable

_WEAK_TYPES = frozenset({"concept-explorer", "generic", ""})

_AFFINITY: list[tuple[re.Pattern[str], str, int]] = [
    (re.compile(r"\bphotosynthesis\b", re.I), "photosynthesis", 90),
    (re.compile(r"\b(respiration|breathing)\b", re.I), "respiration", 85),
    (re.compile(r"\b(seed\s+germinat|sprout)\b", re.I), "seed-germination", 90),
    (re.compile(r"\b(plant\s+growth|growing\s+plant)\b", re.I), "plant-growth", 90),
    (re.compile(r"\b(digestion|digestive)\b", re.I), "digestion", 85),
    (re.compile(r"\b(blood|circulat|heart)\b", re.I), "blood-circulation", 85),
    (re.compile(r"\b(magnet|magnetic)\b", re.I), "magnetism", 90),
    (re.compile(r"\b(electric|circuit|lamp|bulb|conductor|insulator|battery|glow|torch)\b", re.I), "electricity", 85),
    (re.compile(r"\b(reflect|mirror)\b", re.I), "light-reflection", 90),
    (re.compile(r"\b(refract|prism)\b", re.I), "refraction", 90),
    (re.compile(r"\b(acid|base|ph)\b", re.I), "acids-bases", 90),
    (re.compile(r"\b(chemical\s+reaction|burning|magnesium)\b", re.I), "chemical-reaction", 90),
    (re.compile(r"\b(states?\s+of\s+matter|solid|liquid|gas)\b", re.I), "states-of-matter", 80),
    (re.compile(r"\b(heat\s+transfer|conduction)\b", re.I), "heat-transfer", 85),
    (re.compile(r"\b(water\s+cycle)\b", re.I), "water-cycle", 90),
    (re.compile(r"\b(sound|vibration)\b", re.I), "sound", 85),
    (re.compile(r"\b(force|motion|newton)\b", re.I), "force-motion", 85),
    (re.compile(r"\b(solar\s+system|planet)\b", re.I), "solar-system", 90),
    (re.compile(r"\b(human\s+organ|anatomy|organs?\s+and\s+systems?)\b", re.I), "human-organs", 90),
]


def _exp_type(spec_fn: Callable[[str], dict], class_level: str = "CLASS_6") -> str:
    try:
        return str(spec_fn(class_level).get("experimentType") or "")
    except Exception:
        return ""


def score_experiment_rule(
    pattern: re.Pattern[str],
    spec_fn: Callable[[str], dict],
    query: str,
) -> int:
    q = (query or "").strip()
    if not q or not pattern.search(q):
        return -1
    etype = _exp_type(spec_fn)
    score = 40 if etype not in _WEAK_TYPES else 8
    for affinity_re, affinity_type, boost in _AFFINITY:
        if etype == affinity_type and affinity_re.search(q):
            score += boost
    m = pattern.search(q)
    if m:
        score += min(25, len(m.group(0)))
    return score


def pick_best_experiment_rule(
    query: str,
    rules: list[tuple[re.Pattern[str], Callable[[str], dict], str, str, str]],
) -> tuple[Callable[[str], dict], str, str, str] | None:
    best: tuple[int, Callable[[str], dict], str, str, str] | None = None
    for pattern, spec_fn, concept, objective, explanation in rules:
        s = score_experiment_rule(pattern, spec_fn, query)
        if s < 0:
            continue
        if best is None or s > best[0]:
            best = (s, spec_fn, concept, objective, explanation)
    if best is None:
        return None
    _, spec_fn, concept, objective, explanation = best
    return spec_fn, concept, objective, explanation

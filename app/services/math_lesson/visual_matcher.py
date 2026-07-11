"""
Priority-based visualization matcher for Class 1–10 mathematics.

Selects the most specific interactive visualization when multiple topic rules match.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Lower score = weaker / more generic fallback visualizations.
_WEAK_VIZ_TYPES = frozenset({
    "concept-explorer",
    "generic",
    "shapes-basic",
    "probability",
    "bar-model",
})

_VAGUE_PRONOUN_RE = re.compile(r"\b(it|this|that|they|them)\b", re.I)
_TRIANGLE_ANGLE_RE = re.compile(
    r"\b(triangle|triangles)\b.*\bangles?\b|\bangles?\b.*\b(triangle|triangles)\b|"
    r"\b(three|3)\s+angles?\b|"
    r"\binterior\s+angles?\s+(of\s+)?(a\s+)?(triangle|triangles)\b",
    re.I,
)


def build_visualization_query(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Build a query string for visualization matching.
    Merges recent user turns when the current message is vague or omits the shape name.
    """
    q = (query or "").strip()
    if not conversation_history:
        return q

    prior_user: list[str] = []
    for msg in reversed(conversation_history[-8:]):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if content and content != q:
            prior_user.append(content)
        if len(prior_user) >= 3:
            break

    if not prior_user:
        return q

    combined = " ".join([q, *prior_user])
    if _VAGUE_PRONOUN_RE.search(q):
        return combined
    if _TRIANGLE_ANGLE_RE.search(q) and "triangle" not in q.lower():
        if re.search(r"\btriangle", combined, re.I):
            return combined
    return q


# Topic + visualization affinity boosts (query substring checks).
_AFFINITY: list[tuple[re.Pattern[str], str, int]] = [
    (re.compile(r"\bcircles?\b", re.I), "circle", 90),
    (re.compile(r"\bcircles?\b", re.I), "circle-tangent", 85),
    (re.compile(r"\b(tangent|point\s*of\s*contact)\b", re.I), "circle-tangent", 90),
    (re.compile(r"\b(die|dice)\b", re.I), "probability-dice", 90),
    (re.compile(r"\b(coin|toss|flip)\b", re.I), "probability-coin", 85),
    (re.compile(r"\b(even|odd)\s+number\b.*\b(die|dice)\b|\b(die|dice)\b.*\b(even|odd)\b", re.I), "probability-dice", 95),
    (re.compile(r"\b(solve|find\s+(?:the\s+)?(?:value|number)|linear\s+equation|equation)\b", re.I), "linear-graph", 75),
    (re.compile(r"\b\d*x\s*[\+\-]\s*\d+\s*=|[a-z]\s*[\+\-]\s*\d+\s*=", re.I), "linear-graph", 80),
    (re.compile(r"\b(cost|pens?|rupees?|rs\.?|price)\b.*\b(find|solve)\b", re.I), "linear-graph", 70),
    (re.compile(r"\barea\b", re.I), "area-resizer", 75),
    (
        re.compile(
            r"\b(triangle|triangles)\b.*\b(formula|formulas|property|properties|theorem)\b|"
            r"\b(formula|formulas)\b.*\b(triangle|triangles)\b",
            re.I,
        ),
        "triangle-angle-sum",
        92,
    ),
    (re.compile(r"\b(triangle|triangles)\b", re.I), "triangle-angle-sum", 65),
    (re.compile(r"\b(triangle|triangles)\b", re.I), "triangle-explorer", 60),
    (
        re.compile(
            r"\b(triangle|triangles)\b.*\bangles?\b|\bangles?\b.*\b(triangle|triangles)\b|"
            r"\b(three|3)\s+angles?\b|"
            r"\binterior\s+angles?\s+(of\s+)?(a\s+)?(triangle|triangles)\b",
            re.I,
        ),
        "triangle-angle-sum",
        90,
    ),
    (re.compile(r"\bprove\b.*\b180\b|\bangle\s*sums?\b", re.I), "triangle-angle-sum", 85),
    (re.compile(r"\b(mean|median|mode|statistics)\b", re.I), "statistics-lab", 80),
    (re.compile(r"\b(cylinder)\b", re.I), "mensuration-cylinder", 85),
    (re.compile(r"\b(cube)\b", re.I), "mensuration-cube", 85),
    (re.compile(r"\b(transversal|parallel\s+lines?|corresponding)\b", re.I), "parallel-transversal", 85),
    (re.compile(r"\b(factori[sz]e|factori[sz]ation|x\s*[\^²2])\b", re.I), "factor-rectangle", 75),
    (re.compile(r"\b(polynomial|remainder\s+theorem)\b", re.I), "factor-rectangle", 65),
    (re.compile(r"\b((?:a\s*\+\s*b)\s*[\^²2]|algebra\s*tile|2ab)\b", re.I), "algebra-tiles", 85),
    (re.compile(r"\b(irrational|surd|rationali[sz]|sqrt|√|number\s*line)\b", re.I), "number-line", 75),
    (re.compile(r"\b(pythagoras|hypotenuse)\b", re.I), "pythagoras", 85),
    (re.compile(r"\b(quadrilateral|parallelogram|rhombus)\b", re.I), "quadrilateral-morph", 60),
    (re.compile(r"\b(compass|construction|perpendicular\s+bisector)\b", re.I), "geometry-construction", 80),
    (re.compile(r"\b(point|segment|ray)\b.*\b(line|geometry)\b|\belements\s+of\s+geometry\b", re.I), "geometry-basics", 75),
]


def _viz_type(spec_fn: Callable[[], dict[str, Any]]) -> str:
    try:
        return str(spec_fn().get("visualizationType") or "")
    except Exception:
        return ""


def score_visualization_rule(
    pattern: re.Pattern[str],
    spec_fn: Callable[[], dict[str, Any]],
    query: str,
) -> int:
    """Higher score = better match. Returns -1 if pattern does not match."""
    q = (query or "").strip()
    if not q or not pattern.search(q):
        return -1

    vtype = _viz_type(spec_fn)
    score = 40 if vtype not in _WEAK_VIZ_TYPES else 8

    for affinity_re, affinity_type, boost in _AFFINITY:
        if vtype == affinity_type and affinity_re.search(q):
            score += boost

    m = pattern.search(q)
    if m:
        score += min(25, len(m.group(0)))

    return score


def pick_best_visualization_rule(
    query: str,
    rules: list[tuple[re.Pattern[str], Callable[[], dict], str, str, str]],
) -> tuple[Callable[[], dict], str, str, str] | None:
    """Return (spec_fn, concept, objective, explanation) for the best-scoring rule."""
    best: tuple[int, Callable[[], dict], str, str, str] | None = None
    for pattern, spec_fn, concept, objective, explanation in rules:
        s = score_visualization_rule(pattern, spec_fn, query)
        if s < 0:
            continue
        if best is None or s > best[0]:
            best = (s, spec_fn, concept, objective, explanation)
    if best is None:
        return None
    _, spec_fn, concept, objective, explanation = best
    return spec_fn, concept, objective, explanation

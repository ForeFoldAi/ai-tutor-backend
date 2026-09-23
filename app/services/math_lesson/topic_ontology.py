"""
NCERT Class 1–10 topic ontology (v2) — coverage source of truth.

Loads topic_ontology_v2.json (do not regenerate from model knowledge).

Ontology mismatches vs this codebase's conventions (flagged, not silently fixed):
1. `class` is per-class `"1"`…`"10"` in the current fixture (older banded `"1-2"` /
   `"3-5"` rows are still accepted by `_band_matches_class` for back-compat).
2. Several `visualizationType` cells are slash / plus annotations
   (e.g. `"clock-time / money"`, `"algebra-stepper + quadratic-grapher"`), not a
   single type id — resolve_ontology_types() peels these into canonical ids.
3. Parenthetical notes `(new)`, `(grid doubling)`, `(staircase animation)`,
   `(composite scene)`, `(may already be covered…)` are documentation, not types.
4. Meta mentions `constructions-stepper`; product type is `geometry-construction`.
5. Quadratic grapher is a sub-mode of `linear-graph` (`graphMode: quadratic`);
   ontology alias `quadratic-grapher` is accepted as that type.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_ONTOLOGY_PATH = Path(__file__).with_name("topic_ontology_v2.json")

# Strip "(…)" notes from ontology cells.
_PAREN_RE = re.compile(r"\([^)]*\)")

# Aliases → canonical visualizationType registered in the frontend/backend.
_TYPE_ALIASES: dict[str, str] = {
    "quadratic-grapher": "linear-graph",
    "parabola grapher": "linear-graph",
    "constructions-stepper": "geometry-construction",
    "compound-interest-visual (new)": "compound-interest-visual",
    "heights-distances-scene (new)": "heights-distances-scene",
}

# Every type the product can render (SVG + R3F). Keep in sync with TOPIC_RENDERERS
# + generic routers in math-interactive-visualization.tsx.
KNOWN_VISUALIZATION_TYPES: frozenset[str] = frozenset(
    {
        # elementary / shared
        "counting",
        "number-line",
        "place-value",
        "multiplication-grid",
        "bar-model",
        "decimal-blocks",
        "percent-circle",
        "ratio-bar",
        "integer-line",
        "clock-time",
        "money",
        "pattern",
        "matchstick-squares",
        "shape-lab",
        "shapes-basic",
        "symmetry",
        "factor-tree",
        "pythagoras",
        "trig-basic",
        "concept-explorer",
        "fractions",
        "circle",
        "coordinate",
        "graph",
        "probability",
        "generic",
        # topic / advanced
        "algebra-tiles",
        "factor-rectangle",
        "geometry-basics",
        "angle-explorer",
        "parallel-transversal",
        "linear-graph",
        "line-intersection",
        "triangle-explorer",
        "triangle-angle-sum",
        "quadrilateral-morph",
        "statistics-lab",
        "mensuration-cube",
        "mensuration-cylinder",
        "area-resizer",
        "probability-dice",
        "probability-coin",
        "circle-tangent",
        "geometry-construction",
        "sqrt-number-line",
        # Phase extensions
        "algebra-stepper",
        "compound-interest-visual",
        "heights-distances-scene",
        "quadratic-grapher",  # alias accepted; maps to linear-graph quadratic mode
    }
)

# Types that should use the Three.js / R3F path when renderMode is 3d (or forced).
SPATIAL_3D_TYPES: frozenset[str] = frozenset(
    {
        "shape-lab",
        "mensuration-cube",
        "mensuration-cylinder",
        "heights-distances-scene",
    }
)


def resolve_ontology_types(raw: str) -> list[str]:
    """
    Parse an ontology visualizationType cell into canonical type ids.
    Splits on `/` and `+`, strips notes, applies aliases.
    """
    if not (raw or "").strip():
        return []
    cleaned = _PAREN_RE.sub(" ", raw)
    parts = re.split(r"[/+]|,\s*", cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = re.sub(r"\s+", " ", part).strip().lower()
        if not token:
            continue
        # Drop prose fragments that aren't type ids
        if token in {"new", "new sub-mode of linear-graph", "sub-mode of linear-graph"}:
            continue
        if "sub-mode" in token or token.startswith("new "):
            continue
        aliased = _TYPE_ALIASES.get(token, token)
        aliased = aliased.strip()
        if not aliased or aliased in seen:
            continue
        # Keep hyphenated type ids only (allow single words like money, counting)
        if " " in aliased and aliased not in _TYPE_ALIASES:
            # e.g. leftover "parabola grapher" after alias miss
            aliased = _TYPE_ALIASES.get(aliased, aliased.replace(" ", "-"))
        if " " in aliased:
            continue
        seen.add(aliased)
        out.append(aliased)
    return out


@lru_cache(maxsize=1)
def load_topic_ontology() -> dict[str, Any]:
    with _ONTOLOGY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def ontology_topics() -> list[dict[str, Any]]:
    data = load_topic_ontology()
    topics = data.get("topics") or []
    if not isinstance(topics, list):
        raise ValueError("topic_ontology_v2.json: topics must be a list")
    return topics


def primary_visualization_type(topic: dict[str, Any]) -> str | None:
    types = resolve_ontology_types(str(topic.get("visualizationType") or ""))
    return types[0] if types else None



def _band_matches_class(band: str, class_num: int) -> bool:
    b = (band or "").strip()
    if b == "1-2":
        return 1 <= class_num <= 2
    if b == "3-5":
        return 3 <= class_num <= 5
    if b.isdigit():
        return int(b) == class_num
    return True


def pick_ontology_topic(query: str, class_level: str = "") -> dict[str, Any] | None:
    """
    Prefer an ontology row when the query contains the full topic name.
    Word-only fuzzy matching is intentionally avoided so textbook keyword rules
    still win for short problem prompts (Phase 5 template queries include the
    full topic string: "Explain {topic} in {strand}").
    """
    q = (query or "").strip().lower()
    if not q:
        return None
    from app.services.math_lesson.math_tokens import parse_class_num

    class_num = parse_class_num(class_level)
    best: tuple[int, dict[str, Any]] | None = None
    for topic in ontology_topics():
        name = str(topic.get("topic") or "").strip().lower()
        if not name:
            continue
        # Whole-phrase match with word boundaries (avoids "rational" inside "irrational")
        if not re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", q):
            continue
        score = 100 + min(80, len(name))
        if _band_matches_class(str(topic.get("class") or ""), class_num):
            score += 15
        strand = str(topic.get("strand") or "").lower()
        if strand and strand in q:
            score += 5
        if best is None or score > best[0]:
            best = (score, topic)
    if best is None:
        return None
    return best[1]

"""
NCERT Class 1–10 science topic ontology — coverage source of truth.

Loads science_topic_ontology.json (do not regenerate from model knowledge).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

_ONTOLOGY_PATH = __import__("pathlib").Path(__file__).with_name("science_topic_ontology.json")
_PAREN_RE = re.compile(r"\([^)]*\)")

_TYPE_ALIASES: dict[str, str] = {
    "electricity-circuit-lab": "circuit-builder",
    "titration-lab": "acid-base-indicator-lab",
}

KNOWN_VISUALIZATION_TYPES: frozenset[str] = frozenset(
    {
        "human-body-basics",
        "plant-anatomy-lab",
        "animal-habitat-explorer",
        "concept-explorer",
        "food-chain-simple",
        "water-cycle-animator",
        "human-body-system-3d",
        "life-cycle-animator",
        "weather-climate-simulator",
        "simple-machines-lab",
        "states-of-matter-lab",
        "ecosystem-lab",
        "material-sorting-lab",
        "separation-techniques-lab",
        "reaction-simulator",
        "motion-grapher",
        "light-optics-bench",
        "circuit-builder",
        "electricity-circuit-lab",
        "magnet-field-visualizer",
        "microscope-lab",
        "acid-base-indicator-lab",
        "titration-lab",
        "cell-structure-3d",
        "crystal-lattice-3d",
        "combustion-flame-lab",
        "electrolysis-lab",
        "force-pressure-lab",
        "sound-wave-lab",
        "solar-system-3d",
        "molecule-builder-3d",
        "atom-structure-3d",
        "gravitation-orbit-simulator",
        "energy-transformation-lab",
        "tissue-explorer-3d",
        "disease-transmission-simulator",
        "periodic-table-explorer",
        "eye-optics-3d",
        "electromagnet-induction-lab",
        "heredity-punnett-lab",
        # legacy experimentType aliases still accepted
        "photosynthesis",
        "respiration",
        "digestion",
        "blood-circulation",
        "magnetism",
        "electricity",
        "light-reflection",
        "refraction",
        "acids-bases",
        "chemical-reaction",
        "states-of-matter",
        "heat-transfer",
        "water-cycle",
        "sound",
        "force-motion",
        "solar-system",
        "human-organs",
        "plant-growth",
        "seed-germination",
        "generic",
    }
)

LEGACY_TO_CANONICAL: dict[str, str] = {
    "photosynthesis": "plant-anatomy-lab",
    "plant-growth": "plant-anatomy-lab",
    "seed-germination": "life-cycle-animator",
    "respiration": "human-body-system-3d",
    "digestion": "human-body-system-3d",
    "blood-circulation": "human-body-system-3d",
    "human-organs": "human-body-system-3d",
    "magnetism": "magnet-field-visualizer",
    "electricity": "circuit-builder",
    "light-reflection": "light-optics-bench",
    "refraction": "light-optics-bench",
    "acids-bases": "acid-base-indicator-lab",
    "chemical-reaction": "reaction-simulator",
    "states-of-matter": "states-of-matter-lab",
    "heat-transfer": "states-of-matter-lab",
    "water-cycle": "water-cycle-animator",
    "sound": "sound-wave-lab",
    "force-motion": "force-pressure-lab",
    "solar-system": "solar-system-3d",
}


def canonicalize_type(etype: str) -> str:
    t = (etype or "").strip().lower()
    t = _TYPE_ALIASES.get(t, t)
    return LEGACY_TO_CANONICAL.get(t, t)


def resolve_ontology_types(raw: str) -> list[str]:
    if not (raw or "").strip():
        return []
    cleaned = _PAREN_RE.sub(" ", raw)
    parts = re.split(r"[/+]|,\s*", cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = re.sub(r"\s+", " ", part).strip().lower()
        if not token or token in {"new", "factual", "non-graphic", "age-appropriate"}:
            continue
        if "age-appropriate" in token or "non-graphic" in token or "factual" in token:
            continue
        aliased = canonicalize_type(_TYPE_ALIASES.get(token, token))
        if " " in aliased:
            continue
        if aliased in seen:
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
        raise ValueError("science_topic_ontology.json: topics must be a list")
    return topics


def primary_visualization_type(topic: dict[str, Any]) -> str | None:
    types = resolve_ontology_types(str(topic.get("visualizationType") or ""))
    return types[0] if types else None


def _class_matches(band: str, class_num: int) -> bool:
    b = (band or "").strip()
    if b.isdigit():
        return int(b) == class_num
    return True


def pick_ontology_topic(query: str, class_level: str = "") -> dict[str, Any] | None:
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
        if not re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", q):
            # also match significant keyword chunks (≥4 words shortened)
            keywords = [w for w in re.split(r"[^\w]+", name) if len(w) > 4]
            if len(keywords) < 2 or not all(k in q for k in keywords[:2]):
                continue
            score = 40 + min(40, len(name))
        else:
            score = 100 + min(80, len(name))
        if _class_matches(str(topic.get("class") or ""), class_num):
            score += 15
        strand = str(topic.get("strand") or "").lower()
        if strand and strand in q:
            score += 5
        if best is None or score > best[0]:
            best = (score, topic)
    return best[1] if best else None

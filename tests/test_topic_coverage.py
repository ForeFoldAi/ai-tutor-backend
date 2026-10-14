"""CI: every NCERT topic in topic_ontology_v2.json maps to a known visualizationType.

Fails if any topic resolves to zero types, or any resolved type is unregistered.
Also asserts visual_matcher returns a type in that row's acceptable OR-set.

Does not regenerate the ontology — the fixture is the source of truth.

Ontology mismatches vs classLevel conventions are documented in topic_ontology.py.
"""

from __future__ import annotations

from app.services.math_lesson.topic_ontology import (
    KNOWN_VISUALIZATION_TYPES,
    ontology_topics,
    resolve_ontology_types,
)
from app.services.math_lesson.visual_catalog import match_math_visualization


def _class_level_for_band(band: str) -> str:
    """Map ontology class bands to codebase Class N strings (do not rewrite the JSON)."""
    b = (band or "").strip()
    mapping = {
        "1-2": "Class 2",
        "3-5": "Class 4",
        "6": "Class 6",
        "7": "Class 7",
        "8": "Class 8",
        "9": "Class 9",
        "10": "Class 10",
    }
    if b in mapping:
        return mapping[b]
    return f"Class {b}" if b.isdigit() else "Class 6"


def _template_query(topic: dict) -> str:
    strand = str(topic.get("strand") or "").strip()
    name = str(topic.get("topic") or "").strip()
    # Prefer the concrete topic phrase; prepend strand for disambiguation.
    if strand and name:
        return f"Explain {name} in {strand}"
    return f"Explain {name or strand or 'mathematics'}"


def test_ontology_fixture_loads_topics():
    topics = ontology_topics()
    assert len(topics) >= 50, f"expected full Class 1–10 fixture, got {len(topics)} topics"


def test_every_ontology_topic_has_mapped_visualization_type():
    failures: list[str] = []
    for topic in ontology_topics():
        raw = str(topic.get("visualizationType") or "")
        label = f"[{topic.get('class')}] {topic.get('strand')}: {topic.get('topic')}"
        types = resolve_ontology_types(raw)
        if not types:
            failures.append(f"{label} — could not resolve types from {raw!r}")
            continue
        unknown = [t for t in types if t not in KNOWN_VISUALIZATION_TYPES]
        if unknown:
            failures.append(f"{label} — unknown type(s) {unknown} from {raw!r}")

    assert not failures, "Topic coverage gaps:\n" + "\n".join(failures)


def test_every_ontology_topic_resolves_via_matcher():
    """Template query from topic name must land in the row's acceptable type OR-set."""
    failures: list[str] = []
    for topic in ontology_topics():
        raw = str(topic.get("visualizationType") or "")
        acceptable = set(resolve_ontology_types(raw))
        if not acceptable:
            failures.append(f"unresolvable types: {raw!r}")
            continue
        # quadratic-grapher is an alias of linear-graph
        if "quadratic-grapher" in acceptable:
            acceptable.add("linear-graph")
        if "linear-graph" in acceptable:
            acceptable.add("quadratic-grapher")
        query = _template_query(topic)
        class_level = _class_level_for_band(str(topic.get("class") or ""))
        lesson = match_math_visualization(query, class_level)
        got = lesson["visualization"]["visualizationType"]
        label = f"[{topic.get('class')}] {topic.get('topic')}"
        if got not in acceptable:
            failures.append(
                f"{label} — matcher returned {got!r}, expected one of {sorted(acceptable)} "
                f"(query={query!r})"
            )

    assert not failures, "Matcher coverage gaps:\n" + "\n".join(failures)


def test_resolve_ontology_types_handles_slash_and_notes():
    assert resolve_ontology_types("clock-time / money") == ["clock-time", "money"]
    assert "algebra-stepper" in resolve_ontology_types(
        "algebra-stepper + quadratic-grapher"
    )
    assert "linear-graph" in resolve_ontology_types(
        "algebra-stepper + quadratic-grapher"
    )
    assert "algebra-stepper" in resolve_ontology_types(
        "algebra-stepper + parabola grapher (new sub-mode of linear-graph)"
    )
    assert "linear-graph" in resolve_ontology_types(
        "algebra-stepper + parabola grapher (new sub-mode of linear-graph)"
    )
    assert resolve_ontology_types("compound-interest-visual (new)") == ["compound-interest-visual"]
    assert resolve_ontology_types("heights-distances-scene (new)") == ["heights-distances-scene"]
    assert resolve_ontology_types("geometry-construction") == ["geometry-construction"]


def test_new_phase_types_are_registered():
    for t in (
        "algebra-stepper",
        "compound-interest-visual",
        "heights-distances-scene",
        "quadratic-grapher",
        "linear-graph",
    ):
        assert t in KNOWN_VISUALIZATION_TYPES

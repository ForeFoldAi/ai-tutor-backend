"""CI: every NCERT science topic maps to a registered visualizationType."""

from __future__ import annotations

from app.services.science_experiment.experiment_catalog import match_science_experiment
from app.services.science_experiment.topic_ontology import (
    KNOWN_VISUALIZATION_TYPES,
    canonicalize_type,
    ontology_topics,
    resolve_ontology_types,
)
from app.services.science_experiment.visual_catalog import registered_visualization_types


def _class_level(band: str) -> str:
    b = (band or "").strip()
    return f"Class {b}" if b.isdigit() else "Class 6"


def _template_query(topic: dict) -> str:
    strand = str(topic.get("strand") or "").strip()
    name = str(topic.get("topic") or "").strip()
    if strand and name:
        return f"Explain {name} in {strand}"
    return f"Explain {name or strand or 'science'}"


def test_science_ontology_fixture_loads():
    topics = ontology_topics()
    assert len(topics) >= 90, f"expected Class 1–10 science fixture, got {len(topics)}"


def test_every_science_topic_has_registered_type():
    registered = registered_visualization_types() | KNOWN_VISUALIZATION_TYPES
    failures: list[str] = []
    for topic in ontology_topics():
        raw = str(topic.get("visualizationType") or "")
        label = f"[{topic.get('class')}] {topic.get('strand')}: {topic.get('topic')}"
        types = resolve_ontology_types(raw)
        if not types:
            failures.append(f"{label} — could not resolve types from {raw!r}")
            continue
        unknown = [t for t in types if canonicalize_type(t) not in registered and t not in registered]
        if unknown:
            failures.append(f"{label} — unknown type(s) {unknown} from {raw!r}")
    assert not failures, "Science topic coverage gaps:\n" + "\n".join(failures)


def test_every_science_topic_resolves_via_matcher():
    failures: list[str] = []
    for topic in ontology_topics():
        raw = str(topic.get("visualizationType") or "")
        acceptable = {canonicalize_type(t) for t in resolve_ontology_types(raw)}
        if not acceptable:
            failures.append(f"unresolvable: {raw!r}")
            continue
        # Accept aliases that canonicalize into the OR-set
        acceptable |= set(resolve_ontology_types(raw))
        q = _template_query(topic)
        lesson = match_science_experiment(q, _class_level(str(topic.get("class") or "")))
        got = canonicalize_type(str((lesson.get("experiment") or {}).get("experimentType") or ""))
        if got not in acceptable and got not in {canonicalize_type(a) for a in acceptable}:
            failures.append(f"{q!r} → {got!r}, want one of {sorted(acceptable)}")
            continue
        kind = str(topic.get("kind") or "concept")
        exp = lesson.get("experiment") or {}
        if kind == "experiment" and not exp:
            failures.append(f"{q!r} kind=experiment but missing experiment field")
        if kind == "concept" and not exp:
            failures.append(f"{q!r} kind=concept but missing visualization/experiment payload")
    assert not failures, "Science matcher gaps:\n" + "\n".join(failures[:40])

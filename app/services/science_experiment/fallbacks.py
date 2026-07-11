"""Fallback science experiment specs when the LLM omits the JSON block."""

from __future__ import annotations

from typing import Any

from app.services.science_experiment.experiment_catalog import match_science_experiment

_GENERIC_TYPES = frozenset({"concept-explorer", "generic", ""})
_WEAK_LLM_TYPES = frozenset({"concept-explorer", "generic", "", "shapes-basic"})


def build_fallback_science_experiment(query: str, class_level: str = "") -> dict[str, Any]:
    return match_science_experiment(query, class_level)


def get_experiment_catalog_hint(query: str, class_level: str = "") -> str:
    import json

    catalog = build_fallback_science_experiment(query, class_level)
    exp = catalog.get("experiment") or {}
    etype = exp.get("experimentType") or "concept-explorer"
    sliders = exp.get("sliders") or []
    slider_ids = ", ".join(s.get("id", "") for s in sliders if s.get("id")) or "see catalog"
    return (
        "CATALOG SCIENCE EXPERIMENT (mandatory — copy experimentType and slider ids exactly):\n"
        f"- experimentType: {etype}\n"
        f"- title: {exp.get('title', 'Interactive Experiment')}\n"
        f"- slider ids: {slider_ids}\n"
        f"- threeViews: realWorld, microscopic, scientific (all required)\n"
        f"- catalog reference: {json.dumps(exp, ensure_ascii=False)[:500]}"
    )


def merge_catalog_experiment(
    lesson: dict[str, Any] | None,
    catalog: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if catalog is None:
        return lesson
    if lesson is None:
        return dict(catalog)

    cat_exp = catalog.get("experiment") or {}
    cat_type = str(cat_exp.get("experimentType") or "")
    if cat_type in _GENERIC_TYPES:
        return lesson

    llm_exp = lesson.get("experiment") or {}
    llm_type = str(llm_exp.get("experimentType") or "")
    use_catalog = (
        llm_type in _WEAK_LLM_TYPES
        or llm_type != cat_type
        or not llm_exp.get("sliders")
        or not llm_exp.get("threeViews")
        or not str(llm_exp.get("title") or "").strip()
    )
    if not use_catalog:
        return lesson

    merged = dict(lesson)
    merged["experiment"] = cat_exp
    for key in ("conceptName", "learningObjective", "conceptExplanation", "classLevel"):
        if catalog.get(key) and (not lesson.get(key) or llm_type in _WEAK_LLM_TYPES):
            merged[key] = catalog[key]
    return merged

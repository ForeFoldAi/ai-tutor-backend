"""Fallback interactive math lesson specs when the LLM omits the JSON block."""

from __future__ import annotations

from typing import Any

from app.services.math_lesson.visual_catalog import (
    build_universal_math_visual,
    match_math_visualization,
)


def query_requests_animation(query: str) -> bool:
    import re
    return bool(
        re.search(
            r"\b("
            r"animat|visuali[sz]e|interactive|simulation|"
            r"quarter\s+turn|half\s+turn|full\s+turn|"
            r"drag|discover|explorer|tile|canvas|"
            r"show\s+me\s+how|show\s+how"
            r")\b|"
            r"interactive\s+image|generate.*interactive|"
            r"interactive\s+(?:explor|visual|diagram|picture)",
            query or "",
            re.I,
        )
    )


def query_requests_visual_demo(query: str) -> bool:
    """Student wants to see / show something (broader than turn animation)."""
    import re
    return bool(
        re.search(
            r"\b(show\s+me|show\s+us|display|let\s+me\s+see|can\s+you\s+show)\b",
            query or "",
            re.I,
        )
    )


def get_animation_prompt_note(query: str, class_level: str = "") -> str:
    """Catalog-aware animation hint — avoids forcing quarter-turn circle for shape questions."""
    if not query_requests_animation(query) and not query_requests_visual_demo(query):
        return ""
    catalog = build_fallback_math_lesson(query, class_level)
    vtype = str((catalog.get("visualization") or {}).get("visualizationType") or "")
    if vtype == "circle":
        return (
            ' The student asked for ANIMATION — include '
            '"buttons": [{"id": "animate", "label": "▶ Animate Turns", "action": "animate"}] '
            'and visualizationType "circle" with a turnSlider (0–4 quarter turns).'
        )
    return (
        f" The student asked to SEE / SHOW — use visualizationType \"{vtype}\" from the catalog "
        "and include an animate or explore button."
    )


def query_requests_hints(query: str) -> bool:
    import re
    return bool(
        re.search(
            r"\b("
            r"hints?|give\s+me\s+(?:a\s+)?hints?|need\s+(?:a\s+)?hints?|"
            r"progressive\s+hints?|show\s+(?:me\s+)?hints?"
            r")\b",
            query or "",
            re.I,
        )
    )


def query_requests_assessment(query: str) -> bool:
    import re
    return bool(
        re.search(
            r"\b("
            r"assessment|quiz\s+me|test\s+me|conceptual\s+questions?|"
            r"ask\s+me\s+questions?|\d+\s+questions?|give\s+me\s+(?:an?\s+)?(?:assessment|quiz)"
            r")\b",
            query or "",
            re.I,
        )
    )


def query_requests_practice_mode(query: str) -> bool:
    import re
    return bool(
        re.search(
            r"\b(practice\s+mode|practice\s+problems?|practice\s+questions?)\b",
            query or "",
            re.I,
        )
    )


def build_fallback_math_lesson(
    query: str,
    class_level: str = "",
    *,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Synthesize a Class 1–10 math-lesson visualization for every mathematics query.
    Never returns None for a non-empty query.
    """
    q = (query or "").strip()
    if not q:
        return build_universal_math_visual("mathematics", class_level)
    return match_math_visualization(q, class_level, conversation_history=conversation_history)


_GENERIC_VIZ_TYPES = frozenset({"concept-explorer", "generic", ""})
_WEAK_LLM_VIZ_TYPES = frozenset({
    "concept-explorer",
    "generic",
    "",
    "shapes-basic",
    "probability",
    "bar-model",
})


def get_visualization_catalog_hint(
    query: str,
    class_level: str = "",
    *,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Short prompt hint so the LLM uses the catalog-matched visualizationType."""
    import json

    catalog = build_fallback_math_lesson(
        query, class_level, conversation_history=conversation_history
    )
    viz = catalog.get("visualization") or {}
    vtype = viz.get("visualizationType") or "concept-explorer"
    sliders = viz.get("sliders") or []
    slider_ids = ", ".join(s.get("id", "") for s in sliders if s.get("id")) or "see catalog"
    return (
        "CATALOG VISUALIZATION (mandatory — copy visualizationType and slider ids exactly):\n"
        f"- visualizationType: {vtype}\n"
        f"- title: {viz.get('title', 'Interactive Explorer')}\n"
        f"- slider ids: {slider_ids}\n"
        f"- catalog reference: {json.dumps(viz, ensure_ascii=False)[:400]}"
    )


def merge_catalog_visualization(
    lesson: dict[str, Any] | None,
    catalog: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Ensure the lesson uses the best catalog visualization for the query topic."""
    if catalog is None:
        return lesson
    if lesson is None:
        return dict(catalog)

    cat_viz = catalog.get("visualization") or {}
    cat_type = str(cat_viz.get("visualizationType") or "")
    if cat_type in _GENERIC_VIZ_TYPES:
        return lesson

    llm_viz = lesson.get("visualization") or {}
    llm_type = str(llm_viz.get("visualizationType") or "")
    # Always prefer catalog when it has a specific visualization for this topic.
    use_catalog = (
        llm_type in _WEAK_LLM_VIZ_TYPES
        or llm_type != cat_type
        or not llm_viz.get("sliders")
        or not llm_viz.get("buttons")
        or not str(llm_viz.get("title") or "").strip()
    )
    if not use_catalog:
        return lesson

    merged = dict(lesson)
    merged["visualization"] = cat_viz
    for key in ("conceptName", "learningObjective", "conceptExplanation", "classLevel"):
        if catalog.get(key) and (not lesson.get(key) or llm_type in _WEAK_LLM_VIZ_TYPES):
            merged[key] = catalog[key]
    return merged


def prefer_catalog_visualization(
    lesson: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible alias for merge_catalog_visualization."""
    merged = merge_catalog_visualization(lesson, catalog)
    return merged if merged is not None else lesson

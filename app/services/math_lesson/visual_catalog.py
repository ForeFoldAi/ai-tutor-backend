"""
Unified Class 1–10 math visualization matcher.

Every mathematics question should receive a student-friendly interactive visual.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.math_lesson.elementary_catalog import ELEMENTARY_TOPIC_RULES
from app.services.math_lesson.textbook_catalog import _TOPIC_RULES, _lesson, _viz

_CLASS_NUM_RE = re.compile(r"class[_\s]*(\d{1,2})", re.I)


def _parse_class_num(class_level: str) -> int:
    if not class_level:
        return 6
    m = _CLASS_NUM_RE.search(class_level.replace("_", " "))
    if m:
        return max(1, min(10, int(m.group(1))))
    return 6


def _display_level(class_level: str) -> str:
    if class_level:
        return class_level.replace("_", " ")
    return f"Class {_parse_class_num(class_level)}"


def _grade_band_default(class_num: int, class_level: str) -> dict[str, Any]:
    """Grade-appropriate default when no keyword matches."""
    level = _display_level(class_level) if class_level else f"Class {class_num}"
    band: list[tuple[int, int, Callable[[], dict], str, str, str]] = [
        (
            1,
            2,
            lambda: _viz(
                "counting",
                "Let's Count!",
                "Move the slider to count fun objects.",
                sliders=[{"id": "count", "label": "How many?", "min": 0, "max": 15, "step": 1, "default": 5}],
                buttons=[{"id": "animate", "label": "▶ Count", "action": "animate"}],
            ),
            "Counting",
            "Count objects one by one.",
            "Each object gets one number.",
        ),
        (
            3,
            4,
            lambda: _viz(
                "multiplication-grid",
                "Multiplication Explorer",
                "Build arrays with rows and columns.",
                sliders=[
                    {"id": "rows", "label": "Rows", "min": 1, "max": 10, "step": 1, "default": 3},
                    {"id": "cols", "label": "Cols", "min": 1, "max": 10, "step": 1, "default": 4},
                ],
                calcs=[{"id": "product", "label": "Answer", "formula": "rows * cols", "unit": ""}],
            ),
            "Multiplication",
            "Learn times tables visually.",
            "Rows × columns = product.",
        ),
        (
            5,
            5,
            lambda: _viz(
                "fractions",
                "Fraction Explorer",
                "Shade parts of a whole.",
                sliders=[
                    {"id": "numerator", "label": "Parts taken", "min": 0, "max": 8, "step": 1, "default": 2},
                    {"id": "denominator", "label": "Total parts", "min": 1, "max": 8, "step": 1, "default": 4},
                ],
                calcs=[{"id": "pct", "label": "Percent", "formula": "numerator / denominator * 100", "unit": "%"}],
            ),
            "Fractions",
            "Understand parts of a whole.",
            "Numerator ÷ denominator.",
        ),
        (
            6,
            7,
            lambda: _viz(
                "number-line",
                "Number Line",
                "Jump along the line for add and subtract.",
                sliders=[
                    {"id": "start", "label": "Start", "min": -5, "max": 20, "step": 1, "default": 5},
                    {"id": "jump", "label": "Jump", "min": -10, "max": 10, "step": 1, "default": 3},
                ],
                calcs=[{"id": "result", "label": "Answer", "formula": "start + jump", "unit": ""}],
            ),
            "Number Operations",
            "Explore numbers interactively.",
            "Use the number line to add and subtract.",
        ),
        (
            8,
            8,
            lambda: _viz(
                "linear-graph",
                "Graph Explorer",
                "See how y = mx + c works.",
                sliders=[
                    {"id": "m", "label": "Slope m", "min": -3, "max": 3, "step": 0.5, "default": 1},
                    {"id": "c", "label": "Intercept c", "min": -5, "max": 5, "step": 1, "default": 0},
                ],
            ),
            "Graphs",
            "Visualise linear relationships.",
            "Change slope and intercept to move the line.",
        ),
        (
            9,
            10,
            lambda: _viz(
                "concept-explorer",
                "Math Concept Explorer",
                "Use the sliders to explore hands-on.",
                sliders=[
                    {"id": "value1", "label": "Value A", "min": 1, "max": 20, "step": 1, "default": 5},
                    {"id": "value2", "label": "Value B", "min": 1, "max": 20, "step": 1, "default": 3},
                ],
                calcs=[
                    {"id": "sum", "label": "A + B", "formula": "value1 + value2", "unit": ""},
                    {"id": "product", "label": "A × B", "formula": "value1 * value2", "unit": ""},
                ],
                buttons=[{"id": "animate", "label": "▶ Explore", "action": "animate"}],
            ),
            "Mathematics",
            "Explore the concept interactively.",
            "Try changing values and observe.",
        ),
    ]
    for lo, hi, fn, concept, obj, expl in band:
        if lo <= class_num <= hi:
            return _lesson(concept, obj, expl, fn(), level)
    return _lesson("Mathematics Explorer", "Explore with sliders.", "Move sliders and observe.", band[-1][2](), level)


def build_universal_math_visual(query: str, class_level: str = "") -> dict[str, Any]:
    """Always returns a student-friendly visualization for any math query."""
    q = (query or "").strip()
    level = _display_level(class_level)
    title_words = re.sub(r"[^\w\s]", " ", q).split()[:6]
    short_title = " ".join(title_words[:4]).title() or "Math Explorer"
    nums = [int(x) for x in re.findall(r"\b(\d{1,4})\b", q)][:2]
    class_num = _parse_class_num(class_level)
    v1 = nums[0] if nums else min(5 + class_num, 15)
    v2 = nums[1] if len(nums) > 1 else max(2, v1 - 1)

    viz = _viz(
        "concept-explorer",
        f"Explore: {short_title}",
        "Move the sliders and press ▶ Explore. Watch the answers update!",
        sliders=[
            {"id": "value1", "label": "First value", "min": 0, "max": max(20, v1 * 2), "step": 1, "default": v1},
            {"id": "value2", "label": "Second value", "min": 0, "max": max(20, v2 * 2), "step": 1, "default": v2},
        ],
        calcs=[
            {"id": "sum", "label": "Sum (+)", "formula": "value1 + value2", "unit": ""},
            {"id": "diff", "label": "Difference (−)", "formula": "value1 - value2", "unit": ""},
            {"id": "product", "label": "Product (×)", "formula": "value1 * value2", "unit": ""},
        ],
        buttons=[
            {"id": "animate", "label": "▶ Explore", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
    )
    return _lesson(short_title, f"Understand this idea by experimenting.", "Change values and observe.", viz, level)


def match_math_visualization(query: str, class_level: str = "") -> dict[str, Any]:
    """
    Match query to the best Class 1–10 visualization.
    Always returns a lesson (never None) for non-empty queries.
    """
    q = (query or "").strip()
    if not q:
        return build_universal_math_visual("mathematics", class_level)
    level = _display_level(class_level)
    class_num = _parse_class_num(class_level)

    for pattern, spec_fn, concept, objective, explanation in _TOPIC_RULES + ELEMENTARY_TOPIC_RULES:
        if pattern.search(q):
            return _lesson(concept, objective, explanation, spec_fn(), level)

    return _grade_band_default(class_num, class_level)


def match_textbook_visualization(query: str, class_level: str = "") -> dict[str, Any] | None:
    """Backward-compatible: returns None only for empty query."""
    q = (query or "").strip()
    if not q:
        return None
    return match_math_visualization(q, class_level)

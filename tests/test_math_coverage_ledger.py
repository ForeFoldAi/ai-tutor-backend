"""Coverage ledger: ontology remaps, sector, fractions non-steal, deferred labs.

ponytail: assert remaps + dual-signal guards; do not invent deferred viz until traffic warrants.
"""

from __future__ import annotations

from app.services.interactive_templates import extract_typed_params
from app.services.math_lesson.service import finalize_math_answer
from app.services.math_lesson.topic_ontology import (
    KNOWN_VISUALIZATION_TYPES,
    load_topic_ontology,
    ontology_topics,
    resolve_ontology_types,
)
from app.services.math_lesson.visual_catalog import match_math_visualization


DEFERRED = [
    "digit method long division square root",
    "arithmetic progressions staircase",
    "similarity BPT overlay",
    "cone sphere frustum mensuration",
    "open-box net fold animation",
]

# Clear ontology cache so remaps are visible in-process.
load_topic_ontology.cache_clear()


def _topic_types(class_id: str, contains: str) -> list[str]:
    row = next(
        t
        for t in ontology_topics()
        if str(t.get("class")) == class_id and contains.lower() in str(t.get("topic") or "").lower()
    )
    return resolve_ontology_types(str(row.get("visualizationType") or ""))


def test_sqrt_number_line_registered():
    assert "sqrt-number-line" in KNOWN_VISUALIZATION_TYPES


def test_remap_class1_addition_number_line():
    assert _topic_types("1", "Addition and subtraction (single digit")[0] == "number-line"


def test_remap_class3_division_bar_model():
    assert _topic_types("3", "Division introduction")[0] == "bar-model"


def test_remap_class6_algebra_stepper():
    assert _topic_types("6", "Introduction to algebra")[0] == "algebra-stepper"


def test_remap_rational_number_line():
    assert _topic_types("7", "Rational numbers")[0] == "number-line"
    assert _topic_types("8", "Rational numbers (properties)")[0] == "number-line"


def test_remap_class9_circles_chords():
    assert _topic_types("9", "Circles (chords")[0] == "circle"


def test_remap_class10_sector_circle():
    assert _topic_types("10", "Areas related to circles")[0] == "circle"


def test_sector_query_matches_circle():
    lesson = match_math_visualization(
        "Find the area of a sector with radius 7 cm and angle 90 degrees",
        "Class 10",
    )
    assert lesson["visualization"]["visualizationType"] == "circle"
    ids = {s["id"] for s in lesson["visualization"].get("sliders") or []}
    assert "theta" in ids and ("r" in ids or "radius" in ids)


def test_chord_circle_not_tangent():
    lesson = match_math_visualization(
        "Explain angles subtended by a chord in a circle",
        "Class 9",
    )
    assert lesson["visualization"]["visualizationType"] == "circle"


def test_fractions_not_stolen_from_equation():
    vt, _p, pack = extract_typed_params(
        "Solve 2x+5=15. Then x = 10/2 = 5.",
        "math",
    )
    assert pack != "fractions"
    clean, lesson = finalize_math_answer(
        "Solve 2x + 5 = 15.\n2x = 10 so x = 10/2 = 5.",
        "solve the linear equation 2x + 5 = 15",
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] != "fractions"


def test_deferred_list_documented():
    assert len(DEFERRED) >= 4

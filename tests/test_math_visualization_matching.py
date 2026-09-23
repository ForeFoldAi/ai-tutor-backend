"""Production regression tests: math query → correct interactive visualization."""

from __future__ import annotations

import pytest

from app.services.math_lesson.fallbacks import build_fallback_math_lesson, merge_catalog_visualization
from app.services.math_lesson.visual_catalog import match_math_visualization

CLASS_9 = "CLASS_9"


def _vtype(query: str, class_level: str = CLASS_9) -> str:
    return match_math_visualization(query, class_level)["visualization"]["visualizationType"]


# Telangana SCERT Class 9 Mathematics textbook — one expected viz per topic.
TEXTBOOK_CASES: list[tuple[str, str]] = [
    ("number-line", "What are real numbers? Explain with number line."),
    ("sqrt-number-line", "How do we represent sqrt(2) on the number line?"),
    ("sqrt-number-line", "draw a number line and by geometrical construction mark square root 5 on the number line"),
    ("number-line", "What are irrational numbers?"),
    ("algebra-tiles", "Prove (a+b)^2 = a^2 + 2ab + b^2 using algebra tiles."),
    ("factor-rectangle", "Factorise x^2 + 5x + 6"),
    ("factor-rectangle", "What is the remainder theorem?"),
    ("geometry-basics", "What are the basic elements of geometry — point, line segment and ray?"),
    ("parallel-transversal", "A transversal cuts two parallel lines. Explain corresponding angles."),
    ("parallel-transversal", "What are alternate interior angles when a transversal crosses parallel lines?"),
    ("algebra-stepper", "Solve 3x - 5 = 16"),
    ("algebra-stepper", "Solve 2x + 7 = 19"),
    ("algebra-stepper", "A number when added to 7 gives 51. Find the number."),
    ("algebra-stepper", "The cost of 5 pens is Rs 50. Find the cost of one pen."),
    ("triangle-angle-sum", "In a triangle angle A = 50° and angle B = 60°. Find angle C."),
    ("triangle-angle-sum", "can you tell me about the triangle with formula"),
    ("triangle-angle-sum", "can you tell me about triangles and their formulas"),
    ("quadrilateral-morph", "What is a parallelogram? How is it different from a rectangle?"),
    ("area-resizer", "Find the area of a triangle with base 10 cm and height 6 cm."),
    ("area-resizer", "Find the area of a parallelogram with base 8 cm and height 5 cm."),
    ("statistics-lab", "Find mean, median and mode of marks 12, 15, 18, 15, 20."),
    ("mensuration-cylinder", "Find the volume of a cylinder of radius 7 cm and height 10 cm."),
    ("factor-tree", "Find the LCM of 12 and 18"),
    ("factor-tree", "Find HCF of 24 and 36"),
    ("factor-tree", "Prime factorization of 12 and 18"),
    (
        "factor-tree",
        "find the smallest number by which 8788 must be multiplied to obtain a perfect cube",
    ),
    ("mensuration-cube", "Find the surface area and volume of a cube of side 5 cm."),
    ("geometry-construction", "How to construct a perpendicular bisector using compass and ruler?"),
    ("probability-coin", "What is the probability of getting a head when a coin is tossed?"),
    ("probability-dice", "Probability of getting an even number when a die is thrown."),
    ("probability-dice", "A die is rolled 1000 times. What is experimental probability of getting 6?"),
    ("circle", "What is the area of a circle with radius 7 cm?"),
    ("circle", "Explain circumference and diameter of a circle."),
    ("circle-tangent", "What is a tangent to a circle? Explain point of contact."),
]


@pytest.mark.parametrize("expected_vtype,query", TEXTBOOK_CASES)
def test_textbook_visualization_match(expected_vtype: str, query: str):
    assert _vtype(query) == expected_vtype, f"Query: {query}"


def test_circle_never_maps_to_shapes_basic():
    circle_queries = [
        "What is a circle?",
        "Draw a circle with radius 5",
        "Area of circle radius 3",
        "Explain circle geometry",
    ]
    for q in circle_queries:
        v = _vtype(q)
        assert v == "circle", f"Expected circle, got {v} for: {q}"


def test_dice_never_maps_to_generic_probability():
    q = "When a die is thrown once, find probability of an even number."
    v = _vtype(q)
    assert v == "probability-dice"
    assert v != "probability"
    assert v != "concept-explorer"


def test_linear_equation_never_maps_to_concept_explorer():
    linear_queries = [
        "Solve x + 3 = 10",
        "Find the value of x if 4x - 2 = 14",
        "Linear equation in one variable: 5x = 25",
    ]
    for q in linear_queries:
        v = _vtype(q)
        assert v == "algebra-stepper", f"Expected algebra-stepper, got {v} for: {q}"


def test_merge_catalog_overrides_weak_llm_visualization():
    llm_lesson = {
        "conceptName": "Circles",
        "visualization": {
            "visualizationType": "shapes-basic",
            "title": "Shapes",
            "sliders": [{"id": "sides", "label": "Sides", "min": 3, "max": 8, "default": 4}],
        },
    }
    catalog = build_fallback_math_lesson("What is the area of a circle with radius 7 cm?", CLASS_9)
    merged = merge_catalog_visualization(llm_lesson, catalog)
    assert merged is not None
    assert merged["visualization"]["visualizationType"] == "circle"


def test_merge_catalog_overrides_concept_explorer_for_linear():
    llm_lesson = {
        "conceptName": "Equations",
        "visualization": {"visualizationType": "concept-explorer", "title": "Explorer"},
    }
    catalog = build_fallback_math_lesson("Solve 3x - 5 = 16", CLASS_9)
    merged = merge_catalog_visualization(llm_lesson, catalog)
    assert merged is not None
    assert merged["visualization"]["visualizationType"] == "algebra-stepper"


def _locked_shape(query: str) -> str:
    lesson = build_fallback_math_lesson(query, CLASS_9)
    viz = lesson["visualization"]
    objs = viz.get("interactiveObjects") or []
    return str((objs[0] or {}).get("type") or "") if objs else ""


def test_square_locks_to_square_lab_not_polygon_morph():
    for q in (
        "What is a square?",
        "Draw a square",
        "Explain the square shape",
        "Find the area of a square of side 5",
        "A square has how many sides?",
    ):
        assert _vtype(q) == "shape-lab", q
        assert _locked_shape(q) == "square", q
        sliders = build_fallback_math_lesson(q, CLASS_9)["visualization"]["sliders"]
        assert not any(s.get("id") == "sides" for s in sliders), q


def test_square_side_length_taken_from_question():
    lesson = build_fallback_math_lesson("Find the area of a square of side 5", CLASS_9)
    s = next(x for x in lesson["visualization"]["sliders"] if x["id"] == "s")
    assert s["default"] == 5


def test_perfect_square_is_not_a_geometry_square():
    v = _vtype("is 49 a perfect square?")
    assert v != "shape-lab"
    assert v != "shapes-basic"


def test_square_vs_rectangle_stays_comparison():
    assert _vtype("What is the difference between a square and a rectangle?") == "quadrilateral-morph"
    assert _vtype("Compare square and rectangle") == "quadrilateral-morph"


def test_rectangle_locks_without_other_polygons():
    assert _vtype("What is a rectangle?") == "shape-lab"
    assert _locked_shape("What is a rectangle?") == "rectangle"


def test_cube_stays_mensuration_cube():
    assert _vtype("What is a cube?") == "mensuration-cube"
    assert _vtype("Find the surface area and volume of a cube of side 5 cm.") == "mensuration-cube"


def test_merge_catalog_overrides_wrong_locked_shape():
    llm_lesson = {
        "conceptName": "Shapes",
        "visualization": {
            "visualizationType": "shape-lab",
            "title": "Shapes",
            "sliders": [{"id": "sides", "label": "Sides", "min": 3, "max": 8, "default": 4}],
            "buttons": [{"id": "animate", "label": "Draw", "action": "animate"}],
            "interactiveObjects": [{"id": "shape", "type": "pentagon"}],
        },
    }
    catalog = build_fallback_math_lesson("What is a square?", CLASS_9)
    merged = merge_catalog_visualization(llm_lesson, catalog)
    assert merged is not None
    assert merged["visualization"]["visualizationType"] == "shape-lab"
    objs = merged["visualization"]["interactiveObjects"]
    assert objs[0]["type"] == "square"
    assert not any(s.get("id") == "sides" for s in merged["visualization"]["sliders"])

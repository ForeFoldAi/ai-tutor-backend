"""Tests for Class 1–10 unified math visualization catalog."""

from app.services.math_lesson.visual_catalog import match_math_visualization


def test_class1_counting_default():
    lesson = match_math_visualization("Solve this word problem", "Class_1")
    assert lesson["visualization"]["visualizationType"] == "counting"


def test_class1_counting_keyword():
    lesson = match_math_visualization("How many apples are there?", "Class 1")
    assert lesson["visualization"]["visualizationType"] == "counting"


def test_class3_multiplication():
    lesson = match_math_visualization("What is 7 times 8?", "Class 3")
    assert lesson["visualization"]["visualizationType"] == "multiplication-grid"


def test_class5_fractions_default():
    lesson = match_math_visualization("Solve this problem", "Class 5")
    assert lesson["visualization"]["visualizationType"] == "fractions"


def test_class6_number_line():
    lesson = match_math_visualization("Add 12 and 8 on a number line", "Class 6")
    assert lesson["visualization"]["visualizationType"] == "number-line"


def test_class8_linear_graph_default():
    lesson = match_math_visualization("Explain slope", "Class 8")
    assert lesson["visualization"]["visualizationType"] == "linear-graph"


def test_class9_concept_explorer_default():
    lesson = match_math_visualization("Explain polynomials", "Class 9")
    assert lesson["visualization"]["visualizationType"] == "concept-explorer"


def test_quarter_turn_circle():
    lesson = match_math_visualization("What is a quarter turn?", "Class 4")
    assert lesson["visualization"]["visualizationType"] == "circle"


def test_place_value():
    lesson = match_math_visualization("Explain place value of 247", "Class 3")
    assert lesson["visualization"]["visualizationType"] == "place-value"


def test_pythagoras():
    lesson = match_math_visualization("Find hypotenuse using Pythagoras theorem", "Class 10")
    assert lesson["visualization"]["visualizationType"] == "pythagoras"


def test_geometry_rays_segments_lines():
    q = "Let me draw my own lines and identify whether they are rays, segments or lines."
    lesson = match_math_visualization(q, "Class 9")
    assert lesson["visualization"]["visualizationType"] == "geometry-basics"


def test_matchstick_squares_class3():
    lesson = match_math_visualization(
        "Can I make more than one square with the same matchsticks?",
        "Class 3",
    )
    assert lesson["visualization"]["visualizationType"] == "matchstick-squares"


def test_prefer_catalog_over_generic():
    from app.services.math_lesson.fallbacks import prefer_catalog_visualization

    generic = {
        "conceptName": "Math",
        "visualization": {"visualizationType": "concept-explorer", "title": "Generic", "sliders": []},
    }
    catalog = match_math_visualization(
        "draw lines and identify rays segments or lines",
        "Class 9",
    )
    merged = prefer_catalog_visualization(generic, catalog)
    assert merged["visualization"]["visualizationType"] == "geometry-basics"

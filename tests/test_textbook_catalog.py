"""Tests for Class 9 EM textbook visualization catalog."""

from app.services.math_lesson.textbook_catalog import match_textbook_visualization


def test_algebra_tiles_topic():
    lesson = match_textbook_visualization(
        "Explain why (a+b)² not = a²+b². Use algebra tile visualization"
    )
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "algebra-tiles"


def test_factor_rectangle_topic():
    lesson = match_textbook_visualization("Show how to factorise x²+7x+12 using rectangle model")
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "factor-rectangle"


def test_geometry_basics_topic():
    lesson = match_textbook_visualization("difference between point line segment and ray")
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "geometry-basics"


def test_linear_graph_topic():
    lesson = match_textbook_visualization("sliders for y = mx + c graph")
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "linear-graph"


def test_statistics_lab_topic():
    lesson = match_textbook_visualization("enter marks and watch mean median mode bar graph")
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "statistics-lab"


def test_probability_dice_topic():
    lesson = match_textbook_visualization("Roll die 1000 times experimental probability")
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "probability-dice"

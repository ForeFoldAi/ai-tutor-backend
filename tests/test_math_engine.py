"""Tests for SymPy math engine."""

from app.services.math_engine import (
    format_indian,
    try_solve,
)


def test_format_indian():
    assert format_indian(384400) == "3,84,400"
    assert format_indian(365000) == "3,65,000"
    assert format_indian(1200000) == "12,00,000"


def test_moon_travel_problem():
    q = "If you travel 200 km every day, can you reach the Moon in 5 years?"
    ctx = "The average distance from the Earth to the Moon is 3,84,400 km."
    result = try_solve(q, class_level="CLASS_9", chapter_context=ctx)
    assert result is not None
    assert result.solved
    assert result.kind == "travel_distance_comparison"
    assert "3,65,000" in result.final_answer or "365000" in result.final_answer.replace(",", "")
    assert "No" in result.final_answer
    block = result.to_prompt_block()
    assert "MATH ENGINE" in block
    assert "Total Distance = daily distance" in block


def test_linear_equation():
    result = try_solve("Solve 2x + 5 = 15", class_level="CLASS_8")
    assert result is not None
    assert result.solved
    assert "5" in result.final_answer


def test_percentage():
    result = try_solve("What is 20% of 500?", class_level="CLASS_7")
    assert result is not None
    assert result.solved
    assert "100" in result.final_answer


def test_arithmetic():
    result = try_solve("Calculate 25 + 37", class_level="CLASS_4")
    assert result is not None
    assert result.solved
    assert "62" in result.final_answer


def test_fraction_simplify():
    result = try_solve("Simplify 12/18", class_level="CLASS_6")
    assert result is not None
    assert result.solved
    assert "2/3" in result.final_answer


def test_word_problem_linear():
    result = try_solve("A number when added to 7 gives 51. Find the number.", class_level="CLASS_9")
    assert result is not None
    assert result.solved
    assert result.kind == "linear_word_problem"
    assert "44" in result.final_answer


def test_statistics_mean_median_mode():
    result = try_solve(
        "Find mean median mode of marks 12, 15, 18, 15, 20",
        class_level="CLASS_9",
    )
    assert result is not None
    assert result.solved
    assert result.kind == "statistics"
    assert "16" in result.final_answer


def test_cylinder_volume():
    result = try_solve(
        "Find volume of cylinder radius 7 cm height 10 cm",
        class_level="CLASS_9",
    )
    assert result is not None
    assert result.solved
    assert result.kind == "cylinder_volume"


def test_circle_mensuration():
    result = try_solve(
        "What is the area of a circle with radius 7 cm?",
        class_level="CLASS_9",
    )
    assert result is not None
    assert result.solved
    assert result.kind == "circle_mensuration"


def test_triangle_angle_sum():
    result = try_solve(
        "In a triangle angle A = 50 and angle B = 60 find angle C",
        class_level="CLASS_9",
    )
    assert result is not None
    assert result.solved
    assert result.kind == "triangle_angle_sum"
    assert "70" in result.final_answer


def test_probability_dice_even():
    result = try_solve(
        "Probability of getting an even number when a die is thrown",
        class_level="CLASS_9",
    )
    assert result is not None
    assert result.solved
    assert result.kind == "probability_dice"
    assert "1/2" in result.final_answer


def test_cube_root_of_512():
    result = try_solve("what is cure root of 512", class_level="CLASS_8")
    assert result is not None
    assert result.solved
    assert result.kind == "cube_root"
    assert "8" in result.final_answer
    result = try_solve("cube root of 512", class_level="CLASS_8")
    assert result is not None and result.kind == "cube_root"
    assert "8" in result.final_answer


def test_perfect_square_49():
    result = try_solve("is 49 perfect square?", class_level="CLASS_8")
    assert result is not None
    assert result.solved
    assert result.kind == "perfect_square"
    assert "Yes" in result.final_answer
    assert "7" in result.final_answer


def test_cube_root_not_mensuration():
    result = try_solve("Find the cube root of 8", class_level="CLASS_8")
    assert result is not None
    assert result.kind == "cube_root"
    assert "2" in result.final_answer

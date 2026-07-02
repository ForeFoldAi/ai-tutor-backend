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

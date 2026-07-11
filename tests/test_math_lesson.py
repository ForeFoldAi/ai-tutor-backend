"""Tests for interactive math lesson extraction and detection."""

from app.services.math_lesson.fallbacks import build_fallback_math_lesson
from app.services.math_lesson.schemas import MathLesson
from app.services.math_lesson.service import (
    extract_math_lesson_from_answer,
    should_use_interactive_math_lesson,
    strip_math_lesson_block,
)

SAMPLE_LESSON_JSON = """
{
  "conceptName": "Fractions",
  "classLevel": "Class 5",
  "learningObjective": "Understand parts of a whole",
  "conceptExplanation": "A fraction shows equal parts of a whole.",
  "visualization": {
    "visualizationType": "fractions",
    "title": "Pizza Fractions",
    "description": "Drag slices to explore",
    "sliders": [
      {"id": "numerator", "label": "Numerator", "min": 0, "max": 8, "step": 1, "default": 3},
      {"id": "denominator", "label": "Denominator", "min": 1, "max": 8, "step": 1, "default": 4}
    ],
    "liveCalculations": [
      {"id": "percent", "label": "Percentage", "formula": "numerator / denominator * 100", "unit": "%"}
    ],
    "colors": {"primary": "#3B82F6", "secondary": "#10B981", "accent": "#F59E0B", "background": "#F8FAFC", "text": "#1E293B"},
    "studentInteractions": [{"id": "s1", "type": "slide", "description": "Change the numerator", "expectedObservation": "More slices fill"}]
  },
  "guidedExploration": ["What happens if the denominator increases?"],
  "practiceMode": {"easy": "Shade 1/2", "medium": "Add 1/4 + 1/4", "hard": "Compare 2/3 and 3/5", "challenge": "Word problem"},
  "commonMistakes": ["Adding denominators directly"],
  "aiHints": [["Think of equal parts", "Count filled slices", "Numerator over denominator"]],
  "assessment": [
    {"question": "What does the denominator tell us?", "type": "conceptual"},
    {"question": "Is 1/2 greater than 1/4?", "type": "conceptual"},
    {"question": "What is 2/4 in simplest form?", "type": "conceptual"},
    {"question": "If you eat 2 of 8 slices, what fraction remains?", "type": "conceptual"},
    {"question": "Why must fraction parts be equal?", "type": "conceptual"}
  ]
}
"""


def test_strip_math_lesson_block():
    answer = f"**Concept Name**\nFractions\n\n```math-lesson\n{SAMPLE_LESSON_JSON}\n```"
    clean, lesson = strip_math_lesson_block(answer)
    assert "math-lesson" not in clean
    assert "Fractions" in clean
    assert lesson is not None
    assert lesson["conceptName"] == "Fractions"


def test_extract_validates_lesson():
    answer = f"Explain fractions.\n\n```math-lesson\n{SAMPLE_LESSON_JSON}\n```"
    clean, lesson = extract_math_lesson_from_answer(answer)
    assert lesson is not None
    validated = MathLesson.model_validate(lesson)
    assert validated.visualization.visualizationType == "fractions"
    assert len(validated.assessment) == 5
    assert "math-lesson" not in clean


def test_should_use_interactive_for_concept():
    assert should_use_interactive_math_lesson(
        subject_name="Mathematics",
        answer_type="paragraph",
        query="Explain what fractions are",
        question_type="conceptual",
    )


def test_should_use_interactive_for_problem_solving():
    assert should_use_interactive_math_lesson(
        subject_name="Mathematics",
        answer_type="stepwise",
        query="Solve 2x + 5 = 15",
        question_type="problem-solving",
    )


def test_finalize_math_always_returns_lesson_for_math():
    from app.services.math_lesson.service import finalize_math_answer

    clean, lesson = finalize_math_answer(
        "A square has four equal sides.",
        "What is the difference between a square and a rectangle?",
        class_level="CLASS_3",
        subject_name="Mathematics",
    )
    assert clean
    assert lesson is not None
    assert lesson.get("visualization", {}).get("visualizationType")


def test_should_skip_for_greeting():
    assert not should_use_interactive_math_lesson(
        subject_name="Mathematics",
        answer_type="greeting",
        query="Hello",
        question_type="conceptual",
    )


def test_strip_markdown_fence():
    answer = "```markdown\n**To Find**\nExplain angles.\n```"
    clean, _ = strip_math_lesson_block(answer)
    assert "```markdown" not in clean
    assert "**To Find**" in clean


def test_strip_incomplete_math_lesson():
    answer = "**Final Answer**\n90°\n\n```math-lesson\n{\"conceptName\": \"Angles\", \"visualization\": {\"visualizationType\": \"circle\", \"title\": \"Turn\""
    clean, lesson = strip_math_lesson_block(answer)
    assert "math-lesson" not in clean
    assert lesson is None
    assert "Final Answer" in clean


def test_fallback_quarter_turn_lesson():
    lesson = build_fallback_math_lesson(
        "Explain why two quarter turns equal one half turn using animation",
        "CLASS_9",
    )
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "circle"
    assert any(b.get("action") == "animate" for b in lesson["visualization"]["buttons"])


def test_finalize_math_answer_uses_fallback():
    from app.services.math_lesson.service import finalize_math_answer

    answer = "**To Find**\nExplain turns.\n\n**Practice Question**\nTry this."
    clean, lesson = finalize_math_answer(
        answer,
        "two quarter turns equal half turn using animation",
        class_level="CLASS_9",
        subject_name="Mathematics",
    )
    assert lesson is not None
    assert "Practice Question" in clean
    assert lesson["visualization"]["visualizationType"] == "circle"


def test_finalize_math_answer_skips_fallback_when_disabled():
    from app.services.math_lesson.service import finalize_math_answer

    answer = "Let's stay with polynomials in this chapter."
    clean, lesson = finalize_math_answer(
        answer,
        "I will go with option",
        class_level="CLASS_9",
        subject_name="Mathematics",
        allow_fallback=False,
    )
    assert lesson is None
    assert "polynomials" in clean


def test_optional_lesson_fields_stripped_by_default():
    from app.services.math_lesson.service import apply_math_lesson_display_policy

    lesson = {
        "conceptName": "Fractions",
        "aiHints": [["hint"]],
        "assessment": [{"question": "Q?", "type": "conceptual"}],
        "practiceMode": {"easy": "E", "medium": "", "hard": "", "challenge": ""},
    }
    trimmed = apply_math_lesson_display_policy(lesson, "What is a fraction?")
    assert trimmed is not None
    assert trimmed["aiHints"] == []
    assert trimmed["assessment"] == []
    assert trimmed["practiceMode"]["easy"] == ""


def test_optional_lesson_fields_kept_when_requested():
    from app.services.math_lesson.fallbacks import query_requests_assessment, query_requests_hints
    from app.services.math_lesson.service import apply_math_lesson_display_policy

    lesson = {
        "aiHints": [["hint"]],
        "assessment": [{"question": "Q?", "type": "conceptual"}],
    }
    assert query_requests_hints("give me hints please")
    assert query_requests_assessment("quiz me on fractions")
    trimmed = apply_math_lesson_display_policy(lesson, "give me hints and quiz me")
    assert trimmed is not None
    assert trimmed["aiHints"]
    assert trimmed["assessment"]


def test_triangle_three_angles_catalog_match():
    lesson = build_fallback_math_lesson(
        "triangle has three angles",
        "CLASS_9",
    )
    assert lesson["visualization"]["visualizationType"] == "triangle-angle-sum"


def test_triangle_angles_plural_catalog_match():
    lesson = build_fallback_math_lesson(
        "can you tell me about triangles and angles",
        "CLASS_9",
    )
    assert lesson["visualization"]["visualizationType"] == "triangle-angle-sum"


def test_three_angles_with_conversation_context():
    history = [{"role": "user", "content": "What is a triangle?"}]
    lesson = build_fallback_math_lesson(
        "it has three angles and also can you generate the interactive image for the explanation",
        "CLASS_9",
        conversation_history=history,
    )
    assert lesson["visualization"]["visualizationType"] == "triangle-angle-sum"


def test_prefer_catalog_triangle_over_concept_explorer():
    from app.services.math_lesson.fallbacks import merge_catalog_visualization

    llm_lesson = {
        "conceptName": "Triangles",
        "visualization": {
            "visualizationType": "concept-explorer",
            "title": "Math Concept Explorer",
            "sliders": [
                {"id": "value1", "label": "Value A", "min": 1, "max": 20, "default": 5},
                {"id": "value2", "label": "Value B", "min": 1, "max": 20, "default": 3},
            ],
        },
    }
    catalog = build_fallback_math_lesson("triangle has three angles", "CLASS_9")
    merged = merge_catalog_visualization(llm_lesson, catalog)
    assert merged["visualization"]["visualizationType"] == "triangle-angle-sum"


def test_triangle_formula_voice_query():
    lesson = build_fallback_math_lesson(
        "can you tell me about the triangle with formula",
        "CLASS_9",
    )
    assert lesson["visualization"]["visualizationType"] == "triangle-angle-sum"
    assert "180" in lesson["visualization"]["title"] or "Angle" in lesson["visualization"]["title"]


def test_matchstick_squares_catalog_match():
    lesson = build_fallback_math_lesson(
        "Can I make more than one square with the same matchsticks?",
        "CLASS_3",
    )
    vtype = lesson["visualization"]["visualizationType"]
    assert vtype == "matchstick-squares"
    assert "Matchsticks" in lesson["visualization"]["title"] or "Matchstick" in lesson["conceptName"]


def test_prefer_catalog_over_shapes_basic_for_matchsticks():
    from app.services.math_lesson.fallbacks import merge_catalog_visualization

    llm_lesson = {
        "conceptName": "Squares",
        "visualization": {
            "visualizationType": "shapes-basic",
            "title": "Make Squares with Matchsticks",
            "sliders": [{"id": "matchsticks", "label": "Matchsticks", "min": 4, "max": 20, "default": 12}],
        },
    }
    catalog = build_fallback_math_lesson(
        "Can I make more than one square with the same matchsticks?",
        "CLASS_3",
    )
    merged = merge_catalog_visualization(llm_lesson, catalog)
    assert merged["visualization"]["visualizationType"] == "matchstick-squares"

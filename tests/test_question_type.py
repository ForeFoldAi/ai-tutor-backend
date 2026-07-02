"""Question-type detection for pedagogy-aware tutoring."""

from app.services.chat_service import (
    _apply_mathematics_prompt_overrides,
    _is_mathematics_subject,
    _resolve_subject_category,
    _should_use_elementary_math_format,
    detect_question_type,
)


def test_factual_who_when():
    assert detect_question_type("Who invented the telephone?") == "factual"
    assert detect_question_type("When did India gain independence?") == "factual"


def test_conceptual_what_is():
    assert detect_question_type("What is photosynthesis?") == "conceptual"
    assert detect_question_type("Explain the water cycle") == "conceptual"


def test_analytical_why_compare():
    assert detect_question_type("Why does ice float on water?") == "analytical"
    assert detect_question_type("Compare weather and climate") == "analytical"


def test_opinion_discussion():
    assert detect_question_type("Do you think homework helps students?") == "opinion"
    assert detect_question_type("Discuss whether social media is good for teens") == "opinion"


def test_problem_solving():
    assert detect_question_type("Solve 2x + 5 = 15") == "problem-solving"
    assert detect_question_type("Calculate the area of a circle with radius 7") == "problem-solving"
    assert detect_question_type(
        "If you travel 200 km every day, can you reach the Moon in 5 years?"
    ) == "problem-solving"


def test_moon_question_not_personal_dialogue_after_greeting():
    from app.services.chat_service import _build_chat_messages, _is_personal_dialogue_response

    greeting = (
        "Hello Rajesh! Welcome back. I'm your AI Tutor. "
        "What would you like to learn today?"
    )
    question = "If you travel 200 km every day, can you reach the Moon in 5 years?"
    history = [{"role": "assistant", "content": greeting}]
    assert _is_personal_dialogue_response(question, history) is False
    msgs = _build_chat_messages(
        question,
        "Moon distance 3,84,400 km",
        subject_name="Mathematics",
        class_level="CLASS_9",
        conversation_history=history,
    )
    assert "**To Find**" in msgs[0]["content"]
    assert "shared a real-life example" not in msgs[0]["content"]


def test_subject_category_resolution():
    assert _resolve_subject_category("Mathematics") == "Mathematics"
    assert _resolve_subject_category("Social Science - Geography") == "Geography"
    assert _resolve_subject_category("English Literature") == "Language/Literature"
    assert _resolve_subject_category("General Knowledge") is None


def test_is_mathematics_subject():
    assert _is_mathematics_subject("Mathematics") is True
    assert _is_mathematics_subject("Class 8 Algebra") is True
    assert _is_mathematics_subject("Science") is False


def test_mathematics_prompt_eight_section_format():
    instruction, length, interaction, structure, closing = _apply_mathematics_prompt_overrides(
        subject_name="Mathematics",
        answer_type="stepwise",
        heading_scope=None,
        class_level="CLASS_9",
        question_type="problem-solving",
        instruction="plain",
        length_policy="short",
        interaction_policy="generic",
        explanation_structure="",
        user_closing="write now",
    )
    assert "eight-section" in instruction or "mathematics teaching format" in instruction
    assert "100–200 words" in length
    assert "Practice Question" in interaction
    assert "**To Find**" in structure
    assert "**The Formula**" in structure
    assert "Number of buses" in structure
    assert "Never write the word \"or\"" in structure
    assert "Substituting the values" in structure
    assert "Step-by-Step Solution" not in structure
    assert "🎯" not in structure
    assert "all sections in order" in closing


def test_mathematics_prompt_short_answer_skips_template():
    _, length, _, structure, closing = _apply_mathematics_prompt_overrides(
        subject_name="Mathematics",
        answer_type="brief",
        heading_scope=None,
        instruction="plain",
        length_policy="short",
        interaction_policy="generic",
        explanation_structure="full",
        user_closing="write now",
    )
    assert "eight-section template" in length
    assert structure == ""
    assert "essential numbered steps" in closing


def test_non_math_subject_unchanged():
    instruction, length, interaction, structure, closing = _apply_mathematics_prompt_overrides(
        subject_name="Science",
        answer_type="stepwise",
        heading_scope=None,
        instruction="plain",
        length_policy="short",
        interaction_policy="generic",
        explanation_structure="",
        user_closing="write now",
    )
    assert instruction == "plain"
    assert length == "short"
    assert interaction == "generic"
    assert structure == ""
    assert closing == "write now"


def test_class3_concept_math_skips_eight_section():
    from app.services.chat_service import _build_chat_messages

    q = "What is the difference between a square and a rectangle? Show me."
    msgs = _build_chat_messages(
        q,
        "A square has all sides equal. A rectangle has opposite sides equal.",
        subject_name="Mathematics",
        class_level="CLASS_3",
    )
    system = msgs[0]["content"]
    assert "MATHEMATICS ANSWER FORMAT (required for every mathematics question)" not in system
    assert "ELEMENTARY MATHEMATICS ANSWER FORMAT" in system
    assert "Mathematics — elementary" in system
    assert "8-year-old" in system


def test_class9_concept_math_uses_secondary_not_eight_section():
    from app.services.chat_service import _build_chat_messages

    q = "What is the difference between a square and a rectangle? Show me."
    msgs = _build_chat_messages(
        q,
        "A square has all sides equal.",
        subject_name="Mathematics",
        class_level="CLASS_9",
    )
    system = msgs[0]["content"]
    assert "MATHEMATICS ANSWER FORMAT (required for every mathematics question)" not in system
    assert "MATHEMATICS CONCEPT ANSWER FORMAT (Classes 9–12" in system
    assert "14-15-year-old" in system
    assert "ELEMENTARY MATHEMATICS" not in system


def test_class3_problem_solving_keeps_eight_section():
    from app.services.chat_service import _build_chat_messages

    msgs = _build_chat_messages(
        "Ravi has 12 apples and gives 4 away. How many are left?",
        "Subtraction: take away the smaller number.",
        subject_name="Mathematics",
        class_level="CLASS_3",
    )
    assert "**To Find**" in msgs[0]["content"]


def test_should_use_elementary_math_format():
    assert _should_use_elementary_math_format("CLASS_3", "analytical") is True
    assert _should_use_elementary_math_format("CLASS_3", "problem-solving") is False
    assert _should_use_elementary_math_format("CLASS_9", "analytical") is False


def test_animation_note_not_circle_for_shapes():
    from app.services.math_lesson.fallbacks import get_animation_prompt_note

    note = get_animation_prompt_note(
        "What is the difference between a square and a rectangle? Show me.",
        "CLASS_3",
    )
    assert "matchstick-squares" in note or "shapes-basic" in note
    assert "quarter turns" not in note.lower()

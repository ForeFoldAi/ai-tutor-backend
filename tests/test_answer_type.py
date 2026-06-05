"""Answer-type detection and structure tier."""

from app.services.chat_service import (
    _FULL_STRUCTURE_TYPES,
    _structure_tier,
    detect_answer_type,
)


def test_what_is_uses_direct_not_concept():
    assert detect_answer_type("What is weather?") == "short-answer"
    assert _structure_tier("short-answer") == "direct"


def test_explain_what_is_uses_paragraph():
    assert detect_answer_type("Explain what is photosynthesis") == "paragraph"
    assert _structure_tier("paragraph") == "direct"


def test_exam_uses_full_structure():
    assert detect_answer_type("Write a 5 marks answer on democracy") == "exam-format"
    assert detect_answer_type("Write a 5 marks answer on democracy") in _FULL_STRUCTURE_TYPES


def test_brief_and_greeting_compact():
    assert detect_answer_type("Keep it short — what is rain?") == "brief"
    assert detect_answer_type("Hi") == "greeting"
    assert _structure_tier("brief") == "compact"

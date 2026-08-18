"""Answer-type detection and structure tier."""

from app.services.chat_service import (
    _FULL_STRUCTURE_TYPES,
    _structure_tier,
    detect_answer_type,
)


def test_what_is_uses_direct_not_structured():
    assert detect_answer_type("What is weather?") == "short-answer"
    assert _structure_tier("short-answer") == "direct"


def test_what_is_in_detail_uses_structured():
    assert detect_answer_type("What is weather in detail?") == "paragraph"
    assert _structure_tier("paragraph") == "structured"


def test_key_points_uses_bullet_structured():
    assert detect_answer_type("Give key points on weather") == "bullet-points"
    assert _structure_tier("bullet-points") == "structured"


def test_what_kind_of_uses_factual():
    assert detect_answer_type(
        "What kind of materials do we need to make a lamp glow?"
    ) == "factual"
    assert _structure_tier("factual") == "structured"


def test_explain_what_is_uses_paragraph():
    # Response-depth rule: "explain what is X?" should be direct/teacher-like,
    # not a full structured lesson dump.
    assert detect_answer_type("Explain what is photosynthesis") == "short-answer"
    assert _structure_tier("short-answer") == "direct"


def test_exam_uses_full_structure():
    assert detect_answer_type("Write a 5 marks answer on democracy") == "exam-format"
    assert detect_answer_type("Write a 5 marks answer on democracy") in _FULL_STRUCTURE_TYPES


def test_brief_and_greeting_compact():
    assert detect_answer_type("Keep it short — what is rain?") == "brief"
    assert detect_answer_type("Hi") == "greeting"
    assert _structure_tier("brief") == "compact"


def test_strip_embedded_figure_lines():
    from app.services.chat_service import strip_embedded_figure_lines

    raw = (
        "**Temperature**\n"
        "Measured with a thermometer.\n"
        "Fig. 2.5. Temperature\n"
        "Fig. 2.5—Temperature\n"
        "Page 6\n"
        "**Wind**\n"
        "Wind vane shows direction."
    )
    out = strip_embedded_figure_lines(raw)
    assert "Fig." not in out
    assert "Page 6" not in out
    assert "thermometer" in out
    assert "Wind vane" in out


def test_normalize_direct_answer_prose_merges_lone_bold_heading():
    from app.services.chat_service import normalize_direct_answer_prose

    raw = "**Weather**\nis the state of the atmosphere."
    out = normalize_direct_answer_prose(raw)
    assert out.lower().startswith("weather is")
    assert "**" not in out

    broken = "- It happens mainly in the\n\ntroposphere\n, the lowest layer."
    fixed = normalize_direct_answer_prose(broken)
    assert "in the troposphere" in fixed
    assert "in the\n" not in fixed

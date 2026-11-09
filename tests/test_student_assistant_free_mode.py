"""Ask Learn Mastro (free mode) teaches any learning topic instead of redirecting."""

from app.main import app  # noqa: F401 — app import order avoids circular imports
from app.modules.student_assistant.service import _build_messages

_CONTEXT = (
    "STUDENT PROFILE\n- Name: Suneel\n- Grade: 10\n- Curriculum/board: CBSE\n\n"
    "ENROLLED SUBJECTS\n- Mathematics (CBSE, 10): 1/14 chapters\n"
    "  Chapters: Arithmetic Progressions [in_progress]"
)


def _system(mode: str, textbook: str = "") -> str:
    msgs = _build_messages(
        context=_CONTEXT,
        student_name="Suneel",
        query="teach me python programming",
        conversation_history=None,
        agent_mode=mode,
        detected="DETECTED SCOPE:\n- None yet",
        textbook=textbook,
    )
    return msgs[0]["content"]


def test_free_mode_allows_topics_outside_enrolled_subjects():
    system = _system("free")
    assert "programming (Python" in system
    assert "NOT a list of allowed topics" in system
    assert "start teaching it right away" in system
    assert "even if the topic is not one of their subjects" in system


def test_free_mode_forbids_admin_redirect_for_learning_topics():
    system = _system("free")
    assert "Never use this rule for learning or general-knowledge questions" in system
    assert "ask a school admin or tutor" in system


def test_textbook_modes_stay_textbook_only():
    for mode in ("ask", "practice", "explain"):
        system = _system(mode)
        assert "Never refuse a learning topic" not in system

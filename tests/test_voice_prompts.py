"""Phase 5 — voice prompt builders."""

from app.services.voice_prompts import (
    voice_continuation_guidance,
    voice_grade_band,
    voice_turn_type_guidance,
    voice_word_limit,
)


def test_voice_grade_band():
    assert voice_grade_band("CLASS_1") == "1-2"
    assert voice_grade_band("CLASS_3") == "3-5"
    assert voice_grade_band("7") == "6-8"
    assert voice_grade_band("10") == "9-10"


def test_voice_word_limit_greeting_shorter():
    assert voice_word_limit(answer_type="greeting") <= 40
    assert voice_word_limit(expand_deep=True) >= voice_word_limit()


def test_voice_turn_type_guidance_greeting():
    guidance = voice_turn_type_guidance("Hi there!")
    assert "greet" in guidance.lower() or "small talk" in guidance.lower()


def test_voice_prompt_contains_speaking_rules():
    from app.services.voice_prompts import VOICE_SYSTEM_PROMPT

    assert "6–12" in VOICE_SYSTEM_PROMPT
    assert "contractions" in VOICE_SYSTEM_PROMPT.lower()
    assert "exactly what we're studying" in VOICE_SYSTEM_PROMPT
    assert "encyclopedia" in VOICE_SYSTEM_PROMPT.lower()


def test_repeated_question_gets_answer_not_pivot():
    """Student re-asking a question from earlier in the session (not just the
    last turn) should be told to answer it directly, not pivot to new content."""
    history = [
        {"role": "user", "content": "what is political map"},
        {"role": "assistant", "content": "Picture a map of India..."},
        {"role": "user", "content": "delivery partner tracking via google maps"},
        {"role": "assistant", "content": "not covered..."},
        {"role": "user", "content": "can you tell me about the political map"},
        {"role": "assistant", "content": "Imagine a board game..."},
    ]
    guidance = voice_continuation_guidance(
        history, last_assistant="Imagine a board game...", query="what is political map"
    )
    assert "answer it directly again" in guidance.lower()


def test_new_question_not_flagged_as_repeat():
    history = [
        {"role": "user", "content": "what is political map"},
        {"role": "assistant", "content": "Picture a map of India..."},
    ]
    guidance = voice_continuation_guidance(
        history, last_assistant="Picture a map of India...", query="who was Muhammad Ghori"
    )
    assert "answer it directly again" not in guidance.lower()


def test_name_question_does_not_continue_lesson():
    history = [
        {"role": "user", "content": "can you tell me about the chapter"},
        {"role": "assistant", "content": "The chapter asks three big questions..."},
    ]
    guidance = voice_continuation_guidance(
        history,
        last_assistant="The chapter asks three big questions...",
        query="what is your name",
    )
    assert guidance == ""
    from app.services.voice_prompts import build_voice_system_prompt
    from app.services.voice_tutor import TutorState, UnderstandingScores

    system = build_voice_system_prompt(
        "what is your name",
        class_level="CLASS_8",
        subject_name="Social Science",
        chapter="Reshaping India's Political Map",
        student_name="Suneel",
        conversation_history=history,
        tutor_state=TutorState.TEACHING,
        understanding=UnderstandingScores(),
        last_assistant="The chapter asks three big questions...",
        state_guidance="Teach one small idea.",
        understanding_guidance="nod then teach",
        acknowledgment_guidance="",
    )
    assert "do not teach" in system.lower()
    assert "Teach one small idea." not in system

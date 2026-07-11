"""Voice tutor conversational teaching helpers."""

from app.services.voice_tutor import (
    UnderstandingScores,
    TutorState,
    build_acknowledgment_guidance,
    build_voice_mistral_messages,
    evaluate_student_response,
    voice_wants_full_written_answer,
)


def test_voice_wants_full_written_answer():
    assert voice_wants_full_written_answer("show me the full solution")
    assert voice_wants_full_written_answer("give step by step solution")
    assert not voice_wants_full_written_answer("what is photosynthesis?")
    assert not voice_wants_full_written_answer("I got 12 as my answer")


def test_understanding_to_student_hint():
    scores = UnderstandingScores(confusion=0.8)
    assert "simply" in scores.to_student_hint("I don't understand").lower()

    affirm = UnderstandingScores(is_affirmation=True, understanding=0.9)
    assert "Great" in affirm.to_student_hint("yes got it")

    scores = UnderstandingScores()
    assert scores.to_student_hint("why do plants need sunlight") == (
        'I heard: "why do plants need sunlight"'
    )


def test_build_acknowledgment_guidance_problem_attempt():
    scores = UnderstandingScores()
    guidance = build_acknowledgment_guidance("I got 12 as my answer", scores)
    assert "I got 12" in guidance
    assert "attempt" in guidance.lower()


def test_build_voice_messages_natural_voice_rules():
    messages = build_voice_mistral_messages(
        "what is gravity?",
        "Gravity pulls objects together.",
        subject_name="Science",
        tutor_state=TutorState.TEACHING,
        class_level="CLASS_7",
    )
    system = messages[0]["content"]
    assert "6" in system and "12" in system
    assert "contractions" in system.lower()
    assert "exactly what we're studying" in system.lower()
    assert "CLASS 6–8" in system or "6-8" in system
    assert "TTS OUTPUT" in system
    user = messages[-1]["content"]
    assert "imagine" in user.lower() or "friendly teacher" in user.lower()


def test_build_voice_messages_teacher_not_encyclopedia():
    messages = build_voice_mistral_messages(
        "what is photosynthesis?",
        "Plants use sunlight.",
        subject_name="Science",
        tutor_state=TutorState.TEACHING,
        class_level="CLASS_5",
    )
    system = messages[0]["content"]
    assert "encyclopedia" in system.lower()
    assert "let's see" in system.lower() or "think about this" in system.lower()
    assert "process whereby" in system.lower()


def test_build_voice_messages_greeting_turn():
    messages = build_voice_mistral_messages(
        "Hello!",
        "",
        subject_name="Science",
        tutor_state=TutorState.LISTENING,
        class_level="CLASS_3",
    )
    system = messages[0]["content"]
    assert "greet" in system.lower() or "small talk" in system.lower()
    assert "CLASS 3–5" in system or "3-5" in system


def test_build_voice_messages_math_guidance():
    messages = build_voice_mistral_messages(
        "solve 2x + 4 = 10",
        "Linear equations chapter.",
        subject_name="Mathematics",
        tutor_state=TutorState.TEACHING,
    )
    system = messages[0]["content"]
    assert "MATHEMATICS VOICE" in system
    assert "one step" in system.lower()


def test_evaluate_student_response_confusion():
    scores = evaluate_student_response("I'm confused about this step")
    assert scores.confusion >= 0.55

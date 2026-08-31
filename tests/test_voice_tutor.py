"""Voice tutor conversational teaching helpers."""

from unittest.mock import patch

import numpy as np
import pytest

from app.services.voice_tutor import (
    ReplyIntent,
    UnderstandingScores,
    TutorState,
    build_acknowledgment_guidance,
    build_voice_mistral_messages,
    classify_reply_intent,
    classify_understanding_llm,
    evaluate_student_response,
    topic_key,
    update_quiz_state,
    voice_wants_full_written_answer,
)


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(8).astype(np.float32)
    return v / np.linalg.norm(v)


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
    assert "answer directly" in user.lower()
    assert "in everyday terms" in user.lower()  # banned phrase listed in Never say
    assert "friendly ai tutor" in user.lower() or "voice call" in user.lower()


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


def test_evaluate_student_response_dont_know_regex_catches_it():
    """'I don't know' is the single most common DONT_KNOW reply (rule 3(c))
    — it must be recognized by the deterministic regex alone, without
    depending on the BGE fallback (embedding model may be cold)."""
    scores = evaluate_student_response("I don't know")
    assert scores.confusion >= 0.55


@pytest.fixture
def mock_understanding_bge_confusion():
    """Deterministic vectors so an unlisted confusion phrase ('beats me')
    still resolves to the confusion cluster via semantic similarity."""
    prototypes = {"confusion": [_unit_vec(1)], "affirm": [_unit_vec(20)]}
    with patch(
        "app.services.voice_tutor._load_understanding_prototype_embeddings",
        return_value=prototypes,
    ):
        with patch(
            "app.services.vector_service.is_embedding_model_loaded",
            return_value=True,
        ):
            with patch(
                "app.services.image_service.figure_context_bge.embed_query",
                return_value=_unit_vec(1),
            ):
                yield


def test_evaluate_student_response_bge_catches_unlisted_confusion(mock_understanding_bge_confusion):
    """The permanent fix: a confusion phrase absent from _CONFUSION's regex
    list ('beats me') is still caught via BGE semantic similarity, without
    any LLM call and without the regex list needing to enumerate it."""
    scores = evaluate_student_response("beats me")
    assert scores.confusion >= 0.55


@pytest.fixture
def mock_understanding_bge_affirm():
    prototypes = {"confusion": [_unit_vec(1)], "affirm": [_unit_vec(20)]}
    with patch(
        "app.services.voice_tutor._load_understanding_prototype_embeddings",
        return_value=prototypes,
    ):
        with patch(
            "app.services.vector_service.is_embedding_model_loaded",
            return_value=True,
        ):
            with patch(
                "app.services.image_service.figure_context_bge.embed_query",
                return_value=_unit_vec(20),
            ):
                yield


def test_evaluate_student_response_bge_catches_unlisted_affirmation(mock_understanding_bge_affirm):
    scores = evaluate_student_response("all clear on that")
    assert scores.is_affirmation is True
    assert scores.understanding >= 0.7


def test_classify_understanding_llm_parses_response():
    """The background LLM classifier closes the remaining ceiling (negation,
    sarcasm, novel phrasing) that regex + BGE prototypes can't reach."""
    import asyncio

    fake_json = (
        '{"understanding": 0.1, "confusion": 0.9, "is_affirmation": false, '
        '"wants_expansion": false, "wants_quiz": false}'
    )
    with patch("app.services.llm_client.complete", return_value=fake_json):
        result = asyncio.run(
            classify_understanding_llm("if you say so, sure", last_assistant="Does that make sense?")
        )
    assert result is not None
    assert result.confusion >= 0.55
    assert result.is_affirmation is False


def test_classify_understanding_llm_never_raises_on_failure():
    """Fire-and-forget background task — a network/parse failure must return
    None so voice_ws can always fall back to the heuristic, never crash."""
    import asyncio

    async def _boom(*_a, **_k):
        raise RuntimeError("simulated LLM outage")

    with patch("app.services.llm_client.complete", side_effect=_boom):
        result = asyncio.run(classify_understanding_llm("beats me", last_assistant=""))
    assert result is None

    with patch("app.services.llm_client.complete", return_value="not json at all"):
        result2 = asyncio.run(classify_understanding_llm("beats me", last_assistant=""))
    assert result2 is None


def test_classify_reply_intent_closing():
    assert classify_reply_intent("Okay, thank you for your explanation.", quiz_pending=True) == (
        ReplyIntent.CLOSING
    )
    assert classify_reply_intent("thanks!", quiz_pending=False) == ReplyIntent.CLOSING


def test_classify_reply_intent_dont_know_never_confused_with_wrong_answer():
    """Rule 3(c): an 'I don't know' must classify as DONT_KNOW even while a
    quiz is pending — never as a wrong guess."""
    assert classify_reply_intent("I don't know", quiz_pending=True) == ReplyIntent.DONT_KNOW
    assert classify_reply_intent("not sure", quiz_pending=True) == ReplyIntent.DONT_KNOW
    assert classify_reply_intent("I don't know", quiz_pending=False) == ReplyIntent.DONT_KNOW


def test_classify_reply_intent_wrong_answer_only_when_quiz_pending():
    assert classify_reply_intent("I think it's a bird", quiz_pending=True) == (
        ReplyIntent.WRONG_ANSWER
    )
    # Same text with no pending quiz — nothing to be wrong about.
    assert classify_reply_intent("I think it's a bird", quiz_pending=False) == (
        ReplyIntent.NEW_QUESTION
    )


def test_classify_reply_intent_affirmation_is_not_wrong_answer():
    assert classify_reply_intent("yes", quiz_pending=True) == ReplyIntent.NEW_QUESTION


def test_classify_reply_intent_unclear_on_empty_or_garbled():
    assert classify_reply_intent("", quiz_pending=False) == ReplyIntent.UNCLEAR
    assert classify_reply_intent("   ", quiz_pending=False) == ReplyIntent.UNCLEAR


def test_update_quiz_state_first_wrong_attempt_stays_pending():
    pending, attempts = update_quiz_state(
        quiz_pending=True,
        quiz_attempts=0,
        reply_intent=ReplyIntent.WRONG_ANSWER,
        assistant_reply="Not quite — want to try again?",
    )
    assert pending is True
    assert attempts == 1


def test_update_quiz_state_second_wrong_attempt_forces_resolution():
    """Rule 6: never ask a 3rd time — the 2nd wrong/dont-know attempt clears
    the pending question (reply_intent_guidance forces the reveal that turn)."""
    pending, attempts = update_quiz_state(
        quiz_pending=True,
        quiz_attempts=1,
        reply_intent=ReplyIntent.DONT_KNOW,
        assistant_reply="No worries — it was Akbar! Let's move on.",
    )
    assert pending is False
    assert attempts == 0


def test_update_quiz_state_closing_with_pending_quiz_is_resolved():
    """Rule 4: closing out must not silently drop a pending quiz — the
    forced-reveal answer (no trailing '?') clears it."""
    pending, attempts = update_quiz_state(
        quiz_pending=True,
        quiz_attempts=0,
        reply_intent=ReplyIntent.CLOSING,
        assistant_reply="Sounds good! It was sulh-i-kul, Akbar's policy of tolerance.",
    )
    assert pending is False
    assert attempts == 0


def test_update_quiz_state_new_question_starts_tracking():
    pending, attempts = update_quiz_state(
        quiz_pending=False,
        quiz_attempts=0,
        reply_intent=ReplyIntent.NEW_QUESTION,
        assistant_reply="Great question! Do you know what sulh-i-kul means?",
    )
    assert pending is True
    assert attempts == 0


def test_topic_key_strips_question_scaffolding():
    """Question phrasing ('can you tell me about') must not dominate the key —
    only the substantive topic term should remain."""
    assert topic_key("Can you tell me about the Mughals?") == "mughals"
    assert topic_key("") == ""

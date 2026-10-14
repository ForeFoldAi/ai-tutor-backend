"""Unit tests for hybrid conversation intent classifier (regex path)."""

import pytest

from app.services.conversation_intent_classifier import (
    FollowupType,
    IntentClassification,
    answer_type_for_followup,
    classify_followup_intent,
    classify_followup_regex,
    is_clarification_followup,
)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hello", FollowupType.GREETING),
        ("good morning", FollowupType.GREETING),
        ("thank you", FollowupType.SMALL_TALK),
        ("quiz me", FollowupType.GENERATE_QUESTIONS),
        ("give me 5 questions", FollowupType.GENERATE_QUESTIONS),
        ("mcq please", FollowupType.GENERATE_MCQ),
        ("simplify please", FollowupType.SIMPLIFY),
        ("explain in simple words", FollowupType.SIMPLIFY),
        ("summarize this", FollowupType.ASK_SUMMARY),
        ("show me a diagram", FollowupType.ASK_DIAGRAM),
        ("show me the figure", FollowupType.ASK_VISUAL),
        ("give a real life example", FollowupType.ASK_EXAMPLE),
        ("compare solids and liquids", FollowupType.ASK_COMPARISON),
        ("yes", FollowupType.CONTINUE_EXPLANATION),
        ("go on", FollowupType.CONTINUE_EXPLANATION),
        ("go deeper on that", FollowupType.CONTINUE_EXPLANATION),
        ("i did not understand this", FollowupType.CLARIFICATION),
        ("I'm confused", FollowupType.CLARIFICATION),
        ("what is photosynthesis", FollowupType.NEW_TOPIC),
        (
            "give a challenge on this chapter which i can put into implementation",
            FollowupType.NEW_TOPIC,
        ),
        ("What are weather instruments", FollowupType.NEW_TOPIC),
        ("what is your name", FollowupType.SMALL_TALK),
        ("who are you", FollowupType.SMALL_TALK),
        ("what would", FollowupType.SMALL_TALK),
        ("can you tell", FollowupType.SMALL_TALK),
        # Regression: gratitude preceded by an acknowledgment word was
        # falling through to the short-phrase CONTINUE_EXPLANATION fallback
        # and made the tutor re-explain what it had just finished explaining.
        ("thanks for explaining", FollowupType.SMALL_TALK),
        ("right thanks for explaining", FollowupType.SMALL_TALK),
        ("ok thanks", FollowupType.SMALL_TALK),
        ("got it thanks", FollowupType.SMALL_TALK),
        # Regression: a negative reply to a tutor check-in ("...right?") is
        # the same kind of direct answer as "yes" — "no I am not" (4 words)
        # missed the 3-word short-phrase catch-all by one word (unlike the
        # contracted "no I'm not") and fell through to NEW_TOPIC, sending
        # retrieval on the literal reply text and pulling back an unrelated
        # textbook chunk instead of continuing the actual conversation.
        ("no I am not", FollowupType.CONTINUE_EXPLANATION),
        ("no I'm not", FollowupType.CONTINUE_EXPLANATION),
        ("no", FollowupType.CONTINUE_EXPLANATION),
        ("not really", FollowupType.CONTINUE_EXPLANATION),
        ("nope", FollowupType.CONTINUE_EXPLANATION),
        # STT fragments must NOT be keep-teaching merely because they are short.
        ("is", FollowupType.NEW_TOPIC),
        ("same", FollowupType.NEW_TOPIC),
    ],
)
def test_classify_followup_regex(query: str, expected: FollowupType):
    assert classify_followup_regex(query) == expected


def test_thanks_in_long_question_does_not_misfire_as_small_talk():
    """The word-count guard keeps 'thanks to' inside a genuine longer
    question from being misread as a closing remark."""
    q = "why does soil erode, thanks to rainfall patterns over long periods"
    assert classify_followup_regex(q) != FollowupType.SMALL_TALK


def test_clarification_requires_assistant_history():
    history = [
        {"role": "user", "content": "explain shadows"},
        {"role": "assistant", "content": "Place a stick in sunlight..."},
    ]
    assert is_clarification_followup("I don't understand", history)
    assert not is_clarification_followup("I don't understand", None)
    assert not is_clarification_followup("what is science", history)


def test_answer_type_mapping():
    assert answer_type_for_followup(FollowupType.CLARIFICATION) == "clarification"
    assert answer_type_for_followup(FollowupType.SIMPLIFY) == "clarification"
    assert answer_type_for_followup(FollowupType.GREETING) == "greeting"
    assert answer_type_for_followup(FollowupType.SMALL_TALK) == "greeting"
    assert answer_type_for_followup(FollowupType.NEW_TOPIC) is None


def test_classify_followup_intent_returns_classification():
    result = classify_followup_intent("hello")
    assert isinstance(result, IntentClassification)
    assert result.followup_type == FollowupType.GREETING
    assert result.method in ("regex", "bge", "hybrid")
    assert 0.0 <= result.confidence <= 1.0


def test_pronoun_followup_with_history_resolves_via_context():
    from app.services.conversation_context import resolve_conversation_context

    history = [
        {"role": "user", "content": "What is evaporation?"},
        {"role": "assistant", "content": "Evaporation is when liquid turns to vapor..."},
    ]
    ctx = resolve_conversation_context("why does that happen", conversation_history=history)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert "evaporation" in ctx.resolved_topic.lower()

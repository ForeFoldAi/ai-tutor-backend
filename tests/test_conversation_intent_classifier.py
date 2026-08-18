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
    ],
)
def test_classify_followup_regex(query: str, expected: FollowupType):
    assert classify_followup_regex(query) == expected


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

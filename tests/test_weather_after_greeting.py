"""Regression: greeting history must not turn 'what is weather?' into structured paragraph."""

from app.services.chat_service import (
    _resolve_answer_type,
    _structure_tier,
    detect_answer_type,
)
from app.services.conversation_context import resolve_conversation_context
from app.services.conversation_intent_classifier import classify_followup_intent


def test_what_is_weather_after_greeting_is_direct_short_answer():
    q = "what is weather?"
    hist = [
        {
            "role": "assistant",
            "content": (
                "Hi Manith, welcome back. What would you like to learn today "
                "from Chapter 2 - Understanding the Weather?"
            ),
        }
    ]
    cls = classify_followup_intent(q, hist)
    assert cls.followup_type.value == "new_topic"

    conv = resolve_conversation_context(
        q, conversation_history=hist, chapter="Chapter 2 - Understanding the Weather"
    )
    assert conv.followup_type == "new_topic"

    at = _resolve_answer_type(
        q,
        subject_name="Social",
        conversation_history=hist,
        chapter="Chapter 2 - Understanding the Weather",
        conv=conv,
    )
    assert detect_answer_type(q) == "short-answer"
    assert at == "short-answer"
    assert _structure_tier(at) == "direct"

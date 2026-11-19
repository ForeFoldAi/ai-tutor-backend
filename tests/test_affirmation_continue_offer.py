"""Yes-please-explain must continue teaching, not the affirmation choice menu."""

from app.services.chat_service import (
    _is_accepting_tutor_continue_offer,
    _is_affirmation_followup,
    _resolve_answer_type,
)
from app.services.chat_service.dialogue import _continue_offer_prompt_override
from app.services.voice_ack import resolve_dialogue_act

_WEATHER_OFFER = (
    "Weather is the state of the Earth’s atmosphere at a particular time and place. "
    "According to the chapter, weather is shaped by elements like temperature, "
    "precipitation, atmospheric pressure, wind, and humidity.\n\n"
    "Would you like to explore how one of these elements affects daily life?"
)

_LEGITIMATE_GOVT_OFFER = (
    "Take Nannu, a daily wage earner in Delhi. He used an RTI application to get a "
    "duplicate ration card after his was lost.\n\n"
    "- Democratic institutions provide legal tools like RTI.\n"
    "- They create conditions for responsiveness.\n\n"
    "Would you like to see how this connects to the broader concept of a legitimate government?"
)

_CHOICE_MENU = (
    "Great! Now that you know what weather is, would you like a real-life example, "
    "a quick quiz, or to learn how we measure weather — or explore other topics?"
)


def test_yes_please_explain_accepts_offer():
    history = [
        {"role": "user", "content": "what is weather"},
        {"role": "assistant", "content": _WEATHER_OFFER},
    ]
    assert _is_accepting_tutor_continue_offer("Yes please explain", history)
    assert not _is_affirmation_followup("Yes please explain", history)
    assert _resolve_answer_type("Yes please explain", conversation_history=history) == "paragraph"


def test_bare_yes_after_explore_offer_continues():
    history = [
        {"role": "user", "content": "what is weather"},
        {"role": "assistant", "content": _WEATHER_OFFER},
    ]
    assert _is_accepting_tutor_continue_offer("Yes", history)
    assert not _is_affirmation_followup("Yes", history)
    assert _resolve_answer_type("Yes", conversation_history=history) == "paragraph"


def test_bare_yes_after_specific_see_how_offer_continues_despite_ack_act():
    """Screenshot bug: yes to a specific offer must not become the quiz/choice menu."""
    history = [
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": _LEGITIMATE_GOVT_OFFER},
    ]
    assert resolve_dialogue_act("yes") == "ack"
    assert _is_accepting_tutor_continue_offer("yes", history)
    assert not _is_affirmation_followup("yes", history)
    assert (
        _resolve_answer_type("yes", conversation_history=history, dialogue_act="ack")
        == "paragraph"
    )
    override = _continue_offer_prompt_override("yes", history)
    assert override is not None
    assert "choice menu" in override["instruction"].lower() or "quiz" in override["instruction"].lower()
    assert "legitimate government" in override["instruction"].lower()


def test_bare_yes_after_choice_menu_stays_affirmation():
    history = [
        {"role": "user", "content": "what is weather"},
        {"role": "assistant", "content": _CHOICE_MENU},
    ]
    assert not _is_accepting_tutor_continue_offer("Yes", history)
    assert _is_affirmation_followup("Yes", history)
    assert _resolve_answer_type("Yes", conversation_history=history) == "affirmation"
    assert (
        _resolve_answer_type("Yes", conversation_history=history, dialogue_act="ack")
        == "affirmation"
    )


def test_good_thanks_mid_lesson_is_affirmation_not_greeting():
    """Screenshot bug: 'Good thanks' must not reboot with Welcome back."""
    history = [
        {"role": "user", "content": "What is social health"},
        {
            "role": "assistant",
            "content": (
                "Social health means feeling connected with friends and community. "
                "Can you think of one way to improve your social health this week?"
            ),
        },
    ]
    assert _is_affirmation_followup("Good thanks", history)
    assert _resolve_answer_type("Good thanks", conversation_history=history) == "affirmation"
    assert _resolve_answer_type("thanks", conversation_history=history) == "affirmation"


def test_text_stream_yes_to_offer_still_retrieves():
    """Accepted continue-offer must retrieve chapter context (not skip like bare ack)."""
    import asyncio
    from unittest.mock import patch

    from app.services.chat_service import chapter_aware_qa_stream

    called = {"n": 0}

    def fake_retrieve(*_a, **_k):
        called["n"] += 1
        raise RuntimeError("stop-after-retrieve")

    history = [
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": _LEGITIMATE_GOVT_OFFER},
    ]

    async def collect() -> None:
        try:
            async for _t in chapter_aware_qa_stream(
                "yes",
                collection_name="t",
                voice_mode=False,
                conversation_history=history,
            ):
                pass
        except RuntimeError as e:
            if "stop-after-retrieve" not in str(e):
                raise

    with patch("app.services.section_retrieval.retrieve_for_tutor_query", fake_retrieve):
        asyncio.run(collect())
    assert called["n"] == 1

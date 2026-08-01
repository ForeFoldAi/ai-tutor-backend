"""Yes-please-explain must continue teaching, not the affirmation choice menu."""

from app.services.chat_service import (
    _is_accepting_tutor_continue_offer,
    _is_affirmation_followup,
    _resolve_answer_type,
)

_WEATHER_OFFER = (
    "Weather is the state of the Earth’s atmosphere at a particular time and place. "
    "According to the chapter, weather is shaped by elements like temperature, "
    "precipitation, atmospheric pressure, wind, and humidity.\n\n"
    "Would you like to explore how one of these elements affects daily life?"
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


def test_bare_yes_after_choice_menu_stays_affirmation():
    history = [
        {"role": "user", "content": "what is weather"},
        {"role": "assistant", "content": _CHOICE_MENU},
    ]
    assert not _is_accepting_tutor_continue_offer("Yes", history)
    assert _is_affirmation_followup("Yes", history)
    assert _resolve_answer_type("Yes", conversation_history=history) == "affirmation"

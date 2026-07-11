"""Phase 5 — voice prompt builders."""

from app.services.voice_prompts import (
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

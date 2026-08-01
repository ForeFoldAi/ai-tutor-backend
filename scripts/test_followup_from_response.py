"""
Follow-up routing tests based on the tutor's previous response.

Covers the weather flow:
  1) "what is weather" → tutor offers to explore an element
  2) "Yes please explain" → must CONTINUE teaching (paragraph), not the choice menu

Run from ai-tutor-backend:
  .venv/bin/python scripts/test_followup_from_response.py
  .venv/bin/python scripts/test_followup_from_response.py --live   # optional LLM check
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

# ── Sample tutor replies (from real / expected conversation) ─────────────────

WEATHER_ANSWER_WITH_EXPLORE_OFFER = (
    "Weather is the state of the Earth’s atmosphere at a particular time and place. "
    "It describes how hot or cold, wet or dry, calm or windy the air feels around us.\n\n"
    "According to the chapter, weather is shaped by elements like temperature, "
    "precipitation, atmospheric pressure, wind, and humidity. These elements change "
    "constantly, making weather different from day to day.\n\n"
    "Would you like to explore how one of these elements affects daily life?"
)

WRONG_AFFIRMATION_MENU = (
    "Great! Now that you know what weather is, would you like a real-life example, "
    "a quick quiz, or to learn how we measure weather — or explore other topics?"
)

OZONE_ANSWER = (
    "The ozone layer is a special part of the Earth’s atmosphere that protects life "
    "by absorbing most of the sun’s harmful ultraviolet (UV) rays. According to the "
    "chapter, it lies in the stratosphere, above the troposphere where weather happens.\n\n"
    "Would you like to know how the ozone layer is being protected today?"
)

# ── Cases: prior assistant text + student follow-up → expected answer_type ───

CASES: list[dict] = [
    {
        "id": "weather_yes_please_explain",
        "prior_user": "what is weather",
        "prior_assistant": WEATHER_ANSWER_WITH_EXPLORE_OFFER,
        "followup": "Yes please explain",
        "expect_type": "paragraph",
        "expect_accept_offer": True,
        "expect_affirmation": False,
        "note": "Accept explore-offer → keep teaching (not choice menu)",
    },
    {
        "id": "weather_yes_bare",
        "prior_user": "what is weather",
        "prior_assistant": WEATHER_ANSWER_WITH_EXPLORE_OFFER,
        "followup": "Yes",
        "expect_type": "paragraph",
        "expect_accept_offer": True,
        "expect_affirmation": False,
        "note": "Bare yes after single explore offer → continue",
    },
    {
        "id": "weather_sure_go_ahead",
        "prior_user": "what is weather",
        "prior_assistant": WEATHER_ANSWER_WITH_EXPLORE_OFFER,
        "followup": "Sure, go ahead",
        "expect_type": "paragraph",
        "expect_accept_offer": True,
        "expect_affirmation": False,
        "note": "Sure go ahead after explore offer → continue",
    },
    {
        "id": "choice_menu_yes_stays_affirmation",
        "prior_user": "what is weather",
        "prior_assistant": WRONG_AFFIRMATION_MENU,
        "followup": "Yes",
        "expect_type": "affirmation",
        "expect_accept_offer": False,
        "expect_affirmation": True,
        "note": "Bare yes after multi-choice menu → affirmation",
    },
    {
        "id": "choice_menu_quiz_me",
        "prior_user": "what is weather",
        "prior_assistant": WRONG_AFFIRMATION_MENU,
        "followup": "quiz me",
        "expect_type": "quiz",
        "expect_accept_offer": False,
        "expect_affirmation": False,
        "note": "Explicit quiz me → quiz mode",
    },
    {
        "id": "choice_menu_conduct_quiz_not_affirmation",
        "prior_user": "what is weather",
        "prior_assistant": WRONG_AFFIRMATION_MENU,
        "followup": "Conduct quiz",
        "expect_type": "paragraph",
        "expect_accept_offer": False,
        "expect_affirmation": False,
        "note": "Conduct quiz must not hit affirmation menu path",
    },
    {
        "id": "ozone_protection_followup",
        "prior_user": "What is ozone layer?",
        "prior_assistant": OZONE_ANSWER,
        "followup": "How is ozone layer protected?",
        "expect_type": "short-answer",
        "expect_accept_offer": False,
        "expect_affirmation": False,
        "note": "Related follow-up question → answer (not affirmation)",
    },
    {
        "id": "got_it_after_weather",
        "prior_user": "what is weather",
        "prior_assistant": WEATHER_ANSWER_WITH_EXPLORE_OFFER,
        "followup": "Got it",
        "expect_type": "paragraph",
        "expect_accept_offer": True,
        "expect_affirmation": False,
        "note": "Got it after explore offer still accepts continue (short affirm + offer)",
    },
]

_AFFIRMATION_MENU_RE = re.compile(
    r"now that you know|real[- ]?life example|quick quiz|explore other topics",
    re.I,
)
_TEACHES_ELEMENT_RE = re.compile(
    r"\b(temperature|rainfall|precipitation|humidity|wind|pressure|daily life)\b",
    re.I,
)


def _history(prior_user: str, prior_assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": prior_user},
        {"role": "assistant", "content": prior_assistant},
    ]


def run_routing_checks() -> int:
    from app.services.chat_service import (
        _is_accepting_tutor_continue_offer,
        _is_affirmation_followup,
        _resolve_answer_type,
    )

    print("=" * 72)
    print("Follow-up routing (based on prior tutor response)")
    print("=" * 72)

    failed = 0
    for case in CASES:
        hist = _history(case["prior_user"], case["prior_assistant"])
        followup = case["followup"]
        answer_type = _resolve_answer_type(followup, conversation_history=hist)
        accepting = _is_accepting_tutor_continue_offer(followup, hist)
        affirmation = _is_affirmation_followup(followup, hist)

        ok_type = answer_type == case["expect_type"]
        ok_accept = accepting == case["expect_accept_offer"]
        ok_aff = affirmation == case["expect_affirmation"]
        ok = ok_type and ok_accept and ok_aff

        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1

        print(f"\n[{status}] {case['id']}")
        print(f"  note:     {case['note']}")
        print(f"  followup: {followup!r}")
        print(
            f"  type:     got={answer_type!r} expect={case['expect_type']!r}"
            + ("" if ok_type else "  <--")
        )
        print(
            f"  accept:   got={accepting} expect={case['expect_accept_offer']}"
            + ("" if ok_accept else "  <--")
        )
        print(
            f"  affirm:   got={affirmation} expect={case['expect_affirmation']}"
            + ("" if ok_aff else "  <--")
        )

    print("\n" + "-" * 72)
    print(f"Routing: {len(CASES) - failed}/{len(CASES)} passed")
    return failed


async def run_live_check() -> int:
    """Optional: call chapter QA once for the weather → yes please explain turn."""
    from app.services.chat_service import chapter_aware_qa, _resolve_answer_type

    upload_id = os.getenv(
        "TEST_WEATHER_UPLOAD_ID",
        "97ce23be-1c71-45f2-942c-34ea82cc205c",
    )
    collection = os.getenv("TEST_WEATHER_COLLECTION", "CBSE_CLASS_9_Social")
    chapter = os.getenv(
        "TEST_WEATHER_CHAPTER",
        "Chapter 2 - Understanding the Weather",
    )

    history = _history("what is weather", WEATHER_ANSWER_WITH_EXPLORE_OFFER)
    followup = "Yes please explain"
    routed = _resolve_answer_type(followup, conversation_history=history)

    print("\n" + "=" * 72)
    print("Live LLM follow-up check")
    print("=" * 72)
    print(f"  routed answer_type: {routed}")
    if routed != "paragraph":
        print("  FAIL: expected paragraph before LLM call")
        return 1

    try:
        answer, _imgs = await chapter_aware_qa(
            followup,
            collection_name=collection,
            chapter_ids=[upload_id],
            chapter_names=[chapter],
            board="CBSE",
            class_level="CLASS_9",
            subject_name="Social",
            conversation_history=history,
        )
    except Exception as exc:
        print(f"  SKIP live LLM ({exc})")
        return 0

    text = (answer or "").strip()
    print(f"  answer ({len(text)} chars):\n{text[:600]}{'…' if len(text) > 600 else ''}\n")

    looks_like_menu = bool(_AFFIRMATION_MENU_RE.search(text)) and not bool(
        _TEACHES_ELEMENT_RE.search(text)
    )
    teaches = bool(_TEACHES_ELEMENT_RE.search(text))

    if looks_like_menu:
        print("  FAIL: reply looks like affirmation choice menu (no teaching)")
        return 1
    if not teaches:
        print("  WARN: no weather-element keywords — check reply manually")
        return 0

    print("  PASS: reply continues teaching (not choice-only menu)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test follow-ups based on tutor response")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also call chapter_aware_qa for weather → Yes please explain",
    )
    args = parser.parse_args()

    failed = run_routing_checks()
    if args.live:
        failed += asyncio.run(run_live_check())

    print("\n" + "=" * 72)
    if failed:
        print(f"RESULT: {failed} failure(s)")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

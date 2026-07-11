"""
Production conversation scenario matrix (25+ multi-turn cases).

Validates hybrid intent classification, topic inheritance, retrieval query
rewriting, and image gating without calling the LLM.
"""

from __future__ import annotations

import pytest

from app.services.conversation_context import (
    FollowupType,
    ResponseMode,
    VisualIntent,
    resolve_conversation_context,
    should_retrieve_images,
)

# Each scenario: id, query, history, expectations
SCENARIOS: list[dict] = [
    # ── Greetings & small talk ──────────────────────────────────────────────
    {
        "id": "greeting_hello",
        "query": "hello",
        "history": [],
        "followup": FollowupType.GREETING,
        "mode": ResponseMode.GREETING,
        "visual": VisualIntent.NO_VISUALS,
        "images": False,
    },
    {
        "id": "small_talk_thanks",
        "query": "thank you so much",
        "history": [{"role": "user", "content": "what is gravity"}],
        "followup": FollowupType.SMALL_TALK,
        "visual": VisualIntent.NO_VISUALS,
        "images": False,
    },
    # ── New topic / conceptual ──────────────────────────────────────────────
    {
        "id": "new_topic_photosynthesis",
        "query": "What is photosynthesis?",
        "history": [],
        "followup": FollowupType.NEW_TOPIC,
        "mode": ResponseMode.EXPLANATION,
        "images": True,
    },
    {
        "id": "new_topic_weather_instruments",
        "query": "What are the weather instruments?",
        "history": [],
        "followup": FollowupType.NEW_TOPIC,
        "images": True,
    },
    {
        "id": "new_topic_challenge_request",
        "query": "give a challenge on this chapter which i can put into implementation",
        "history": [],
        "followup": FollowupType.NEW_TOPIC,
        "retrieval_contains": ["challenge"],
    },
    # ── Clarification (the reported bug) ────────────────────────────────────
    {
        "id": "clarification_shadow_challenge",
        "query": "i did not understand this challenge",
        "history": [
            {
                "role": "user",
                "content": "give a challenge on this chapter which i can put into implementation",
            },
            {
                "role": "assistant",
                "content": (
                    "According to the chapter, try this hands-on challenge: Observe how shadows "
                    "change during the day by placing a stick upright in sunlight."
                ),
            },
        ],
        "followup": FollowupType.CLARIFICATION,
        "mode": ResponseMode.FOLLOWUP,
        "visual": VisualIntent.NO_VISUALS,
        "images": False,
        "retrieval_contains": ["shadow", "challenge"],
    },
    {
        "id": "clarification_explain_again",
        "query": "can you explain that again",
        "history": [
            {"role": "user", "content": "explain eclipses"},
            {"role": "assistant", "content": "An eclipse happens when the Moon blocks the Sun..."},
        ],
        "followup": FollowupType.CLARIFICATION,
        "retrieval_contains": ["eclipse"],
        "images": False,
    },
    {
        "id": "clarification_confused",
        "query": "I'm still confused about this",
        "history": [
            {"role": "user", "content": "how does a pinhole camera work"},
            {"role": "assistant", "content": "Light travels in straight lines through a tiny hole..."},
        ],
        "followup": FollowupType.CLARIFICATION,
        "images": False,
    },
    {
        "id": "clarification_lost_me",
        "query": "you lost me on that part",
        "history": [
            {"role": "user", "content": "what is refraction"},
            {"role": "assistant", "content": "Refraction bends light when it enters a new medium."},
        ],
        "followup": FollowupType.CLARIFICATION,
        "images": False,
    },
    # ── Continue / affirm short replies ─────────────────────────────────────
    {
        "id": "continue_yes",
        "query": "yes",
        "history": [{"role": "user", "content": "What is evaporation?"}],
        "followup": FollowupType.CONTINUE_EXPLANATION,
        "resolved_contains": ["evaporation"],
        "images": False,
    },
    {
        "id": "continue_go_on",
        "query": "go on",
        "history": [
            {"role": "user", "content": "tell me about the water cycle"},
            {"role": "assistant", "content": "Water evaporates from oceans..."},
        ],
        "followup": FollowupType.CONTINUE_EXPLANATION,
        "resolved_contains": ["water cycle"],
        "images": False,
    },
    {
        "id": "continue_go_deeper",
        "query": "go deeper on that",
        "history": [
            {"role": "user", "content": "what are states of matter"},
            {"role": "assistant", "content": "Matter exists as solid, liquid, and gas."},
        ],
        "followup": FollowupType.CONTINUE_EXPLANATION,
        "images": False,
    },
    {
        "id": "pronoun_why_that",
        "query": "why does that happen",
        "history": [
            {"role": "user", "content": "What is condensation?"},
            {"role": "assistant", "content": "Condensation forms water droplets on cool surfaces."},
        ],
        "followup": FollowupType.CONTINUE_EXPLANATION,
        "resolved_contains": ["condensation"],
    },
    {
        "id": "pronoun_what_about_it",
        "query": "what about it",
        "history": [
            {"role": "user", "content": "explain magnets"},
            {"role": "assistant", "content": "Magnets attract iron objects."},
        ],
        "followup": FollowupType.CONTINUE_EXPLANATION,
        "resolved_contains": ["magnets"],
    },
    # ── Simplify ──────────────────────────────────────────────────────────────
    {
        "id": "simplify_request",
        "query": "explain in simpler words",
        "history": [
            {"role": "user", "content": "what is cellular respiration"},
            {"role": "assistant", "content": "Cells break down glucose to release energy..."},
        ],
        "followup": FollowupType.SIMPLIFY,
        "mode": ResponseMode.FOLLOWUP,
        "resolved_contains": ["simpler"],
        "images": False,
    },
    {
        "id": "simplify_eli5",
        "query": "eli5 please",
        "history": [
            {"role": "user", "content": "how do vaccines work"},
            {"role": "assistant", "content": "Vaccines train the immune system..."},
        ],
        "followup": FollowupType.SIMPLIFY,
        "images": False,
    },
    # ── Quiz / MCQ ────────────────────────────────────────────────────────────
    {
        "id": "quiz_followup",
        "query": "give me 5 questions",
        "history": [
            {"role": "user", "content": "Explain the Thar Desert climate"},
            {"role": "assistant", "content": "The Thar Desert is arid and hot..."},
        ],
        "followup": FollowupType.GENERATE_QUESTIONS,
        "mode": ResponseMode.QUIZ,
        "resolved_contains": ["thar", "desert"],
        "images": False,
    },
    {
        "id": "mcq_followup",
        "query": "mcq on this topic",
        "history": [
            {"role": "user", "content": "types of rocks"},
            {"role": "assistant", "content": "Igneous, sedimentary, and metamorphic..."},
        ],
        "followup": FollowupType.GENERATE_MCQ,
        "mode": ResponseMode.MCQ,
        "images": False,
    },
    # ── Summary ─────────────────────────────────────────────────────────────
    {
        "id": "summary_request",
        "query": "summarize in short",
        "history": [
            {"role": "user", "content": "explain the nitrogen cycle"},
            {"role": "assistant", "content": "Nitrogen moves between air, soil, and living things..."},
        ],
        "followup": FollowupType.ASK_SUMMARY,
        "mode": ResponseMode.SUMMARY,
        "resolved_contains": ["nitrogen"],
    },
    # ── Examples ────────────────────────────────────────────────────────────
    {
        "id": "example_request",
        "query": "show me a real life example",
        "history": [
            {"role": "user", "content": "what is friction"},
            {"role": "assistant", "content": "Friction opposes motion between surfaces."},
        ],
        "followup": FollowupType.ASK_EXAMPLE,
        "mode": ResponseMode.FOLLOWUP,
        "retrieval_contains": ["example", "friction"],
    },
    # ── Visual / diagram ────────────────────────────────────────────────────
    {
        "id": "visual_figure",
        "query": "show me the figure from the chapter",
        "history": [
            {"role": "user", "content": "parts of a flower"},
            {"role": "assistant", "content": "A flower has petals, stamen, and pistil."},
        ],
        "followup": FollowupType.ASK_VISUAL,
        "mode": ResponseMode.VISUAL_LEARNING,
        "visual": VisualIntent.REQUIRED_VISUALS,
        "images": True,
    },
    {
        "id": "diagram_request",
        "query": "draw a diagram of the water cycle",
        "history": [],
        "followup": FollowupType.ASK_DIAGRAM,
        "visual": VisualIntent.REQUIRED_VISUALS,
        "images": True,
    },
    # ── Comparison ────────────────────────────────────────────────────────────
    {
        "id": "comparison_request",
        "query": "compare weather and climate",
        "history": [],
        "followup": FollowupType.ASK_COMPARISON,
        "mode": ResponseMode.FOLLOWUP,
    },
    {
        "id": "comparison_followup",
        "query": "what is the difference between them",
        "history": [
            {"role": "user", "content": "solids and liquids"},
            {"role": "assistant", "content": "Solids have fixed shape; liquids take container shape."},
        ],
        "followup": FollowupType.ASK_COMPARISON,
        "resolved_contains": ["solids", "liquids"],
    },
    # ── Multi-turn chains ─────────────────────────────────────────────────────
    {
        "id": "chain_learn_then_clarify",
        "query": "I don't get the stick part",
        "history": [
            {"role": "user", "content": "give a hands-on activity for shadows"},
            {"role": "assistant", "content": "Place a stick upright and mark shadows every hour."},
            {"role": "user", "content": "okay"},
            {"role": "assistant", "content": "Would you like to predict tomorrow's shadow positions?"},
        ],
        "followup": FollowupType.CLARIFICATION,
        "retrieval_contains": ["shadow", "stick"],
        "images": False,
    },
    {
        "id": "chain_learn_then_quiz",
        "query": "test me now",
        "history": [
            {"role": "user", "content": "what is the solar system"},
            {"role": "assistant", "content": "The solar system has the Sun and eight planets."},
        ],
        "followup": FollowupType.GENERATE_QUESTIONS,
        "resolved_contains": ["solar"],
        "images": False,
    },
    {
        "id": "chain_explain_then_simplify",
        "query": "make it easier",
        "history": [
            {"role": "user", "content": "how does photosynthesis work"},
            {"role": "assistant", "content": "Plants use chlorophyll to convert light into glucose."},
        ],
        "followup": FollowupType.SIMPLIFY,
        "images": False,
    },
    {
        "id": "chain_explain_then_example",
        "query": "give an example from daily life",
        "history": [
            {"role": "user", "content": "what is inertia"},
            {"role": "assistant", "content": "Objects resist changes in motion."},
        ],
        "followup": FollowupType.ASK_EXAMPLE,
        "retrieval_contains": ["inertia", "example"],
    },
    {
        "id": "chain_explain_then_summary",
        "query": "briefly summarize",
        "history": [
            {"role": "user", "content": "structure of the atom"},
            {"role": "assistant", "content": "Atoms have protons, neutrons, and electrons."},
        ],
        "followup": FollowupType.ASK_SUMMARY,
        "resolved_contains": ["atom"],
    },
    {
        "id": "chain_explain_then_visual",
        "query": "can I see the diagram",
        "history": [
            {"role": "user", "content": "human digestive system"},
            {"role": "assistant", "content": "Food passes through mouth, stomach, intestines."},
        ],
        "followup": FollowupType.ASK_DIAGRAM,
        "visual": VisualIntent.REQUIRED_VISUALS,
        "images": True,
    },
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_conversation_scenario(scenario: dict):
    ctx = resolve_conversation_context(
        scenario["query"],
        conversation_history=scenario.get("history"),
    )

    assert ctx.followup_type == scenario["followup"].value, (
        f"Expected {scenario['followup']}, got {ctx.followup_type} "
        f"(method={ctx.intent_method}, conf={ctx.intent_confidence})"
    )

    if "mode" in scenario:
        assert ctx.response_mode == scenario["mode"]

    if "visual" in scenario:
        assert ctx.visual_intent == scenario["visual"]

    if "images" in scenario:
        assert should_retrieve_images(ctx, chapter_ids=["ch1"]) == scenario["images"]

    for token in scenario.get("resolved_contains", []):
        assert token.lower() in ctx.resolved_topic.lower(), (
            f"Expected '{token}' in resolved_topic={ctx.resolved_topic!r}"
        )

    for token in scenario.get("retrieval_contains", []):
        assert token.lower() in ctx.retrieval_query.lower(), (
            f"Expected '{token}' in retrieval_query={ctx.retrieval_query!r}"
        )


def test_scenario_count_at_least_25():
    assert len(SCENARIOS) >= 25


def test_intent_debug_metadata_present():
    ctx = resolve_conversation_context("hello")
    debug = ctx.to_debug()
    assert "intent_method" in debug
    assert "intent_confidence" in debug
    assert debug["intent_method"] in ("regex", "bge", "hybrid")

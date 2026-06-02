"""Tests for conversation context and visual retrieval gating."""

from app.services.conversation_context import (
    FollowupType,
    ResponseMode,
    VisualIntent,
    resolve_conversation_context,
    should_retrieve_images,
)
from app.services.image_service.symbolic_image_filters import level3_overlap_allowed
from app.services.image_service.image_intent_extractor import ImageIntent
from app.services.tts_sanitize import sanitize_for_tts


def test_greeting_no_images():
    ctx = resolve_conversation_context("hello")
    assert ctx.followup_type == FollowupType.GREETING.value
    assert ctx.visual_intent == VisualIntent.NO_VISUALS
    assert not should_retrieve_images(ctx, chapter_ids=["ch1"])


def test_quiz_followup_inherits_topic_not_images():
    history = [
        {"role": "user", "content": "Explain the Thar Desert climate"},
        {"role": "assistant", "content": "The Thar Desert is arid..."},
    ]
    ctx = resolve_conversation_context("give me 5 questions", conversation_history=history)
    assert ctx.followup_type == FollowupType.GENERATE_QUESTIONS.value
    assert "thar" in ctx.resolved_topic.lower() or "desert" in ctx.resolved_topic.lower()
    assert not should_retrieve_images(ctx, chapter_ids=["ch1"])


def test_explain_topic_allows_images():
    ctx = resolve_conversation_context("Explain the Thar Desert in detail")
    assert ctx.response_mode == ResponseMode.EXPLANATION
    assert should_retrieve_images(ctx, chapter_ids=["ch1"])


def test_visual_request_required():
    ctx = resolve_conversation_context("Show me a diagram of photosynthesis")
    assert ctx.visual_intent == VisualIntent.REQUIRED_VISUALS
    assert should_retrieve_images(ctx, chapter_ids=["ch1"])


def test_yes_continue_no_images():
    history = [{"role": "user", "content": "What is evaporation?"}]
    ctx = resolve_conversation_context("yes", conversation_history=history)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert not should_retrieve_images(ctx, chapter_ids=["ch1"])


def test_level3_overlap_strict():
    intent = ImageIntent(
        query="thar desert",
        core_concept="thar desert",
        required_terms=["thar", "desert"],
        supporting_terms=[],
        negative_terms=[],
        query_type="concept_explanation",
        preferred_types=[],
        excluded_types=[],
        concept_tokens=frozenset({"thar", "desert", "climate"}),
        entities=["thar"],
        rag_section_tokens=frozenset(),
        requested_visuals=False,
    )
    assert not level3_overlap_allowed(intent, frozenset({"weather"}))
    assert level3_overlap_allowed(intent, frozenset({"thar", "desert"}))


def test_tts_strips_markdown():
    out = sanitize_for_tts("**Photosynthesis** uses *light*.")
    assert "**" not in out
    assert "Photosynthesis" in out

"""Tests for vision caption weak-caption detection and formatting."""

from app.services.image_service.vision_caption_service import (
    format_educational_vision_caption,
    is_weak_text_caption,
)


def test_weak_generic_chapter_caption():
    assert is_weak_text_caption("Illustration showing chapter 1 - the ever-evolving world of science")


def test_weak_ocr_fragment():
    assert is_weak_text_caption(
        "Illustration — Ah, but wha the wall or a wrist watch tells us the tim get prepared to go t"
    )


def test_strong_caption_not_weak():
    assert not is_weak_text_caption("Fig. 2.11 AWS at a glacial lake of Sikkim")


def test_minimal_fig_label_is_weak():
    assert is_weak_text_caption("Fig. 2.2")


def test_format_educational_vision_caption():
    out = format_educational_vision_caption(
        "a sunflower with roots in the soil",
        subject="Science",
        chapter_title="Chapter 1 - The Ever-Evolving World of Science",
    )
    assert out.startswith("a sunflower")
    assert "Science" in out


def test_distinctive_single_term_passes_required_gate():
    from app.services.image_service.image_intent_extractor import ImageIntent
    from app.services.image_service.symbolic_image_filters import count_required_term_matches

    intent = ImageIntent(
        query="sunflower plant roots",
        core_concept="sunflower plant roots",
        required_terms=["sunflower plant roots", "sunflower"],
        supporting_terms=[],
        negative_terms=[],
        query_type="concept_explanation",
        preferred_types=[],
        excluded_types=[],
        concept_tokens=frozenset({"sunflower", "plant", "roots"}),
        entities=[],
        rag_section_tokens=frozenset(),
        requested_visuals=False,
    )
    matches, passes = count_required_term_matches(
        intent,
        "a sunflower with roots and roots. science textbook figure.",
    )
    assert "sunflower" in matches
    assert passes

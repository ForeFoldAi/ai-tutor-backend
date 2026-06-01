"""Unit tests for topic-centric figure grounding (mandatory / supporting stages)."""

from unittest.mock import MagicMock, patch

from app.services.figure_grounding import (
    RetrievalStages,
    assemble_payload_rows,
    caption_conflicts_with_topic,
    classify_mandatory_figure,
    compute_supporting_score,
    FigureRankRecord,
)
from app.services.image_intent_extractor import extract_image_intent
from app.services.symbolic_image_filters import level3_overlap_allowed
from app.services.textbook_image_extraction import build_figure_context, parse_figure_slots


def test_parse_figure_slots_extracts_nearby_text():
    page = (
        "The Thar Desert is a large arid region. "
        "Fig. 2.3. Sand dunes in Rajasthan. "
        "These dunes shift with wind."
    )
    slots = parse_figure_slots(page)
    assert len(slots) == 1
    assert "2.3" in slots[0]["figure_number"]
    assert "Thar" in slots[0]["nearby_text_before"] or "arid" in slots[0]["nearby_text_before"]
    assert "dunes" in slots[0]["nearby_text_after"].lower() or "wind" in slots[0]["nearby_text_after"].lower()


def test_build_figure_context_combines_sections():
    ctx = build_figure_context(
        caption="Fig. 1.1. Test diagram",
        nearby_before="Students observe the process.",
        section_title="Weather Systems",
        chapter_title="Climate",
    )
    assert "Weather Systems" in ctx
    assert "Students observe" in ctx


def test_level3_single_token_blocked():
    intent = extract_image_intent("Explain Thar Desert")
    assert not level3_overlap_allowed(intent, frozenset({"desert"}))
    assert level3_overlap_allowed(intent, frozenset({"thar", "desert"}))


def test_caption_conflict_rejects_off_topic_caption():
    intent = extract_image_intent("Explain Thar Desert")
    im = MagicMock()
    im.caption = "Indian Peacock in the forest"
    im.page_text_snippet = ""
    reject, reason = caption_conflicts_with_topic(intent, im, concept_specificity=5.0)
    assert reject
    assert "caption" in reason


def test_classify_mandatory_on_rag_figure_reference():
    intent = extract_image_intent("Explain sand dunes")
    im = MagicMock()
    im.figure_number = "2.3"
    im.caption = "Fig. 2.3. Sand dunes"
    im.page_text_snippet = ""
    im.textbook_upload_id = "u1"
    im.page_index = 5
    im.section_title = "Deserts"
    im.has_caption = True
    rag = [MagicMock(page_content="See Fig. 2.3 for the dune pattern.", metadata={})]
    is_mand, topic_anchor, reason = classify_mandatory_figure(
        intent, im, rag,
        context_score=70.0, section_score=60.0, page_proximity=10.0, concept_specificity=50.0,
    )
    assert is_mand
    assert "educational_content_reference" in reason


def test_mandatory_bypasses_supporting_rank():
    """Mandatory images are assembled first without competing on weighted score."""
    im_m = MagicMock()
    im_m.id = "mand-1"
    im_m.file_name = "anchor.jpg"
    im_m.caption = ""
    rec_m = FigureRankRecord(
        image_id="mand-1", file_name="anchor.jpg", mandatory=True,
        topic_anchor=True, figure_context_score=72.0, final_score=72.0,
        selected_reason="mandatory_stage:topic_anchor_figure",
    )
    im_s = MagicMock()
    im_s.id = "sup-1"
    im_s.file_name = "support.jpg"
    im_s.caption = "Supporting chart"
    rec_s = FigureRankRecord(
        image_id="sup-1", file_name="support.jpg",
        figure_context_score=40.0, caption_score=35.0, section_score=30.0,
        type_score=80.0, role_score=75.0, page_score=10.0, clip_score=0.1,
    )
    compute_supporting_score(rec_s)

    stages = RetrievalStages(
        mandatory_images=[(im_m, rec_m)],
        supporting_images=[(im_s, rec_s)],
    )
    rows = assemble_payload_rows(stages)
    assert len(rows) == 2
    assert rows[0][0].file_name == "anchor.jpg"
    assert rows[0][1] == 72.0
    assert rows[1][0].file_name == "support.jpg"
    assert rows[1][1] == rec_s.final_score
    assert rec_s.final_score < 1000.0


def test_page_background_plate_rejects_full_page_raster():
    from app.services.textbook_image_extraction import _is_page_background_plate, reject_figure_rect

    assert _is_page_background_plate(2480, 3508)
    assert not _is_page_background_plate(1894, 1894)
    assert not _is_page_background_plate(750, 393)
    rejected, reason = reject_figure_rect(0, 0, 595, 842, 595, 842)
    assert rejected and reason == "area_gt_70pct"


def test_minimal_caption_detected():
    from app.services.figure_context_gates import is_minimal_figure_caption

    assert is_minimal_figure_caption("Fig. 2.2")
    assert not is_minimal_figure_caption("Fig. 2.4.2. Cloudy weather")


def test_supporting_score_weights():
    rec = FigureRankRecord(
        image_id="x", file_name="x.jpg",
        figure_context_score=100.0, caption_score=100.0, section_score=100.0,
        type_score=100.0, role_score=100.0, page_score=100.0, clip_score=100.0,
    )
    total = compute_supporting_score(rec)
    assert total == 100.0

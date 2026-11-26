"""Vision-LLM figure captions: parsing, validation, storing, display."""

from types import SimpleNamespace

from app.services.image_service.figure_caption_vlm import (
    FigureCaption,
    FigureContext,
    _rate_limit_wait,
    apply_to_row,
    audience,
    build_messages,
    parse_response,
    validation_problems,
)
from app.services.image_service.pdf_figure_context import resolve_display_caption
from app.services.pdf_extract_pipeline.figure_pairing import FIGURE_LABEL_RE


def _row(**kw):
    base = dict(
        figure_number=None, caption=None, generated_caption=None, caption_source="none",
        title=None, visual_summary=None, figure_context="", educational_tags=None,
        is_decorative=False, educational_role="unknown", context_bge_indexed=True,
        multimodal_indexed=True, page_index=0, chapter_title=None, upload=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_parse_plain_and_wrapped_json():
    raw = ('<think>looking</think>Here: {"is_teaching_figure": true, "kind": "diagram", "title": "BPT",'
           ' "caption": "This image shows triangle ABC with DE parallel to BC.", "labels_seen": ["A","B","C","D","E"]}}')
    r = parse_response(raw)
    assert r is not None and r.kind == "diagram" and r.is_teaching_figure
    assert r.caption.startswith("Triangle ABC")  # filler stripped, capitalised
    assert parse_response("no json here") is None


def test_unknown_kind_defaults_to_illustration():
    r = parse_response('{"kind": "sketch", "caption": "A boy reading a letter under a tree."}')
    assert r.kind == "illustration" and r.is_teaching_figure


def test_validation_rejects_invented_points_and_measurements():
    ctx = FigureContext(printed_caption="Fig. 6.14", text_before="AB || DC and EF || AB")
    good = FigureCaption(True, "diagram", "Trapezium", "Trapezium ABCD with EF parallel to AB.", ["A", "B", "C", "D", "E", "F"])
    assert validation_problems(good, ctx) == []
    stray = FigureCaption(True, "diagram", "x", "Trapezium ABCD with point G on EF.", ["A", "B", "C", "D", "E", "F"])
    assert any("unseen points" in p for p in validation_problems(stray, ctx))
    invented = FigureCaption(True, "diagram", "x", "Trapezium ABCD whose side AB is 7 cm long.", ["A", "B", "C", "D"])
    assert any("measurement" in p for p in validation_problems(invented, ctx))
    seen = FigureCaption(True, "diagram", "x", "Circle with radius 5 cm.", ["5cm", "O"])
    assert validation_problems(seen, FigureContext()) == []


def test_validation_ignores_capitals_in_photos():
    ctx = FigureContext(printed_caption="Fig. 5 – Chauri Chaura, 1922.")
    r = FigureCaption(True, "photo", "x", "A crowd at Chauri Chaura in Gorakhpur, 1922.", [])
    assert validation_problems(r, ctx) == []


def test_apply_teaching_caption():
    row = _row(figure_number="6.14", caption="Fig. 6.14")
    apply_to_row(row, FigureCaption(True, "diagram", "Trapezium with a parallel line",
                                     "Trapezium ABCD with EF parallel to AB.", ["A", "B"]))
    assert row.caption_source == "vlm" and row.generated_caption.startswith("Trapezium")
    assert row.title == "Trapezium with a parallel line"
    assert row.visual_summary.startswith("type: diagram")
    assert not row.context_bge_indexed and not row.multimodal_indexed
    assert not row.is_decorative


def test_apply_hides_unnumbered_mascot_but_flags_numbered_text_crop():
    mascot = _row()
    apply_to_row(mascot, FigureCaption(False, "cartoon", "Mascot", "A cartoon notebook.", []))
    assert mascot.is_decorative and mascot.educational_tags.startswith("junk:vlm_not_teaching")

    bad_crop = _row(figure_number="6.31", caption="Fig. 6.31")
    apply_to_row(bad_crop, FigureCaption(False, "text", "Text", "Partial text fragments.", []))
    assert not bad_crop.is_decorative
    assert bad_crop.caption_source == "none" and bad_crop.educational_tags == "needs_recrop"


def test_display_prefers_vlm_caption():
    row = _row(figure_number="6.14", caption="Fig. 6.14", caption_source="vlm",
               generated_caption="Trapezium ABCD with EF parallel to AB.")
    assert resolve_display_caption(row) == "Trapezium ABCD with EF parallel to AB."


def test_prompt_mentions_class_and_printed_caption():
    msgs = build_messages("data:image/jpeg;base64,xx", FigureContext(class_level="CLASS_10", printed_caption="Fig. 3 – Dyer"))
    assert "Class 10 student" in msgs[0]["content"]
    assert "Printed caption: Fig. 3 – Dyer" in msgs[1]["content"][0]["text"]
    assert audience("") == "a school student"


def test_rate_limit_wait_uses_retry_after():
    exc = SimpleNamespace(response=SimpleNamespace(status_code=429, headers={"retry-after": "20"}))
    assert _rate_limit_wait(exc, 0) == 20.0
    assert _rate_limit_wait(SimpleNamespace(response=SimpleNamespace(status_code=400, headers={})), 0) is None
    assert _rate_limit_wait(exc, 99) is None


def test_figure_label_regex_accepts_figure_word():
    assert FIGURE_LABEL_RE.search("Figure 9.1 Schematic").group(1) == "9.1"
    assert FIGURE_LABEL_RE.search("Fig.9.2").group(1) == "9.2"
    assert FIGURE_LABEL_RE.search("configure 3 things") is None

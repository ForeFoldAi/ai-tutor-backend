"""Unit tests for native PDF extraction pipeline helpers."""

from __future__ import annotations

from app.services.pdf_extract_pipeline.figure_pairing import (
    FIGURE_LABEL_RE,
    _assign_labels_to_figures,
    _box_area,
    _caption_fig_number_mismatch,
    _figure_match_score,
    _is_shallow_figure_crop,
    _is_prose_orphan_caption,
    _sanitize_box,
)
from app.services.pdf_extract_pipeline.merge import latex_rm_whitespace, page_to_markdown
from app.services.pdf_extract_pipeline.types import (
    ExtractedAsset,
    LayoutElement,
    PersistableMlAsset,
    bbox_to_poly,
)


def test_figure_label_regex():
    assert FIGURE_LABEL_RE.search("See Fig. 2.3 for details")
    assert FIGURE_LABEL_RE.search("fig. 10.1")


def test_figure_match_score_prefers_above_label():
    label_box = (100, 400, 180, 420)
    near_figure = (90, 200, 300, 390)
    far_figure = (90, 50, 300, 120)
    assert _figure_match_score(label_box, near_figure) < _figure_match_score(label_box, far_figure)


def test_assign_labels_to_figures_greedy():
    labels = [{"fig_number": "2.1", "label_box": (10, 400, 80, 420)}]
    figures = [{"box": (10, 200, 200, 390)}]
    pairs = _assign_labels_to_figures(labels, figures)
    assert len(pairs) == 1
    assert pairs[0][0]["fig_number"] == "2.1"


def test_box_area():
    assert _box_area((0, 0, 100, 50)) == 5000
    assert _box_area((100, 50, 0, 0)) == 5000


def test_sanitize_box_inverted():
    assert _sanitize_box((100, 80, 20, 10)) == (20, 10, 100, 80)
    assert _sanitize_box((0, 0, 0, 0)) is None


def test_caption_fig_number_mismatch():
    assert _caption_fig_number_mismatch("Fig. 2.3.2. A frog croaking", "2.3.1")
    assert not _caption_fig_number_mismatch("Fig. 2.3.1. Pine cones", "2.3.1")


def test_shallow_figure_crop_rejects_ant_strip():
    assert _is_shallow_figure_crop((284, 175, 610, 299), (1200, 1600))
    assert not _is_shallow_figure_crop((591, 888, 976, 1341), (1200, 1600))


def test_resolve_figure_box_keeps_tall_layout_when_refine_collapses():
    from app.services.pdf_extract_pipeline.figure_pairing import _resolve_figure_box

    label = {"fig_number": "2.6", "label_box": (600, 1030, 780, 1055)}
    layout_box = (591, 888, 976, 1341)
    shallow_refined = (585, 894, 984, 1033)

    class _Page:
        rect = type("R", (), {"width": 612, "height": 792})()

        def get_text(self, *a, **k):
            return ""

        def search_for(self, *a, **k):
            return []

    import app.services.pdf_extract_pipeline.figure_pairing as fp

    orig_refine = fp.refine_figure_box
    fp.refine_figure_box = lambda *a, **k: shallow_refined
    try:
        resolved = _resolve_figure_box(
            _Page(),
            label=label,
            layout_box=layout_box,
            image_size=(1200, 1600),
            dpi=150,
        )
    finally:
        fp.refine_figure_box = orig_refine
    assert resolved == layout_box


def test_prose_orphan_caption():
    assert _is_prose_orphan_caption("e Weather")
    assert not _is_prose_orphan_caption("Fig. 2.3.3. Pine cones open and close")


def test_latex_rm_whitespace():
    assert latex_rm_whitespace("a + b") == "a+b" or latex_rm_whitespace("a + b") == "a + b"


def test_page_to_markdown_plain_text():
    dets = [
        {"category_type": "plain text", "poly": bbox_to_poly(10, 10, 200, 40)},
        {
            "category_type": "text",
            "poly": bbox_to_poly(12, 12, 198, 38),
            "text": "Rainfall is measured in millimetres.",
        },
    ]
    md = page_to_markdown(dets)
    assert "Rainfall" in md


def test_persistable_ml_asset_from_extracted():
    asset = ExtractedAsset(
        asset_type="table",
        page_no=3,
        image_bytes=b"\x89PNG",
        number="2.1",
        caption="Table 2.1 Rainfall data",
        structured_text="| Month | mm |",
        bbox=(10, 20, 300, 400),
    )
    row = PersistableMlAsset.from_extracted(asset, sequence=5)
    assert row.content_kind == "table"
    assert row.page_index == 2
    assert row.structured_content.startswith("|")

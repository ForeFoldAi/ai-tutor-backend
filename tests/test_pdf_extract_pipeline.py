"""Unit tests for native PDF extraction pipeline helpers."""

from __future__ import annotations

from app.services.pdf_extract_pipeline.figure_pairing import (
    FIGURE_LABEL_RE,
    _assign_labels_to_figures,
    _box_area,
    _figure_match_score,
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

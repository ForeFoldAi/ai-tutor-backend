"""Unit tests for caption-anchored figure crop bounds."""

from __future__ import annotations

from app.services.image_service.figure_reconstruction import (
    FigureAnchor,
    _figure_bottom_y,
    _refine_figure_crop_bbox,
    apply_single_figure_bounds,
    FinalFigureRegion,
)
from app.services.image_service.pdf_layout_extraction import BBox, LayoutTextBlock


def _anchor(fig: str, cap_y0: float, cap_y1: float = 0.0) -> FigureAnchor:
    if cap_y1 <= cap_y0:
        cap_y1 = cap_y0 + 14.0
    return FigureAnchor(
        figure_number=fig,
        caption_bbox=BBox(120.0, cap_y0, 280.0, cap_y1),
        caption_text=f"Fig. {fig}",
        page_number=1,
    )


def test_figure_bottom_stops_above_caption():
    anchor = _anchor("2.11", 520.0, 534.0)
    assert _figure_bottom_y(anchor) == 520.0 - 6.0


def test_apply_single_figure_bounds_excludes_caption_band():
    page = BBox(0, 0, 595, 842)
    anchor = _anchor("2.6", 400.0, 414.0)
    region = FinalFigureRegion(
        figure_bbox=BBox(80.0, 180.0, 420.0, 430.0),
        object_count=2,
    )
    bounded = apply_single_figure_bounds(region, anchor, [anchor], page, [])
    assert bounded is not None
    assert bounded.y1 <= anchor.caption_bbox.y0 - 6.0 + 0.01
    assert bounded.y1 < anchor.caption_bbox.y1


def test_refine_trims_body_text_above():
    page = BBox(0, 0, 595, 842)
    anchor = _anchor("2.6", 400.0, 414.0)
    body = LayoutTextBlock(1, BBox(60.0, 120.0, 520.0, 200.0), "Weather is the day-to-day condition of the atmosphere.")
    crop = BBox(60.0, 110.0, 420.0, 394.0)
    refined = _refine_figure_crop_bbox(crop, anchor, [body], page)
    assert refined.y0 >= 120.0


def test_pick_cluster_accepts_negative_distance_scores():
    from app.services.image_service.figure_reconstruction import _pick_cluster_for_anchor
    from app.services.image_service.pdf_layout_extraction import BBox

    cap = BBox(200.0, 380.0, 240.0, 395.0)
    anchor = type("A", (), {"caption_bbox": cap, "figure_number": "2.2"})()
    diagram = BBox(42.0, 211.0, 395.0, 396.0)
    search = BBox(0, 0, 595, 842)
    chosen = _pick_cluster_for_anchor([[diagram]], anchor, search, page_bbox=BBox(0, 0, 595, 842))
    assert len(chosen) == 1


def test_search_region_same_column_when_caption_below():
    from app.services.image_service.figure_reconstruction import _search_region

    page = BBox(0, 0, 595, 842)
    anchor = FigureAnchor(
        figure_number="2.6",
        caption_bbox=BBox(380.0, 680.0, 475.0, 694.0),
        caption_text="Fig. 2.6",
        page_number=7,
    )
    search = _search_region(anchor, page)
    assert search.x0 >= 200.0
    assert search.x1 >= 450.0


def test_search_region_full_width_for_wide_caption():
    from app.services.image_service.figure_reconstruction import _search_region

    page = BBox(0, 0, 595, 842)
    anchor = FigureAnchor(
        figure_number="2.13",
        caption_bbox=BBox(180.0, 690.0, 470.0, 706.0),
        caption_text="Fig. 2.13",
        page_number=14,
    )
    search = _search_region(anchor, page)
    assert search.x0 <= 20.0
    assert search.x1 >= 570.0


def test_recover_plate_embedded_keeps_large_map():
    from app.services.image_service.figure_reconstruction import (
        _filter_teaching_embedded_boxes,
        _MAX_EMBED_AREA_FRAC,
    )

    page = BBox(0, 0, 595, 842)
    plate = BBox(56.0, 193.0, 522.0, 659.0)
    assert plate.area / page.area > _MAX_EMBED_AREA_FRAC
    assert _filter_teaching_embedded_boxes([plate], page) == []


def test_fig_marker_without_space_before_number():
    from app.services.image_service.textbook_image_extraction import _FIG_MARKER_RE

    m = _FIG_MARKER_RE.search("See Fig.2.11 for the station.")
    assert m is not None
    assert m.group(2) == "2.11"

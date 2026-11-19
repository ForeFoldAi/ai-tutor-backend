"""Tests for shared figure layout filters."""

from app.services.image_service.figure_filters import (
    boxes_overlap_ratio,
    figure_asset_priority,
    is_near_duplicate_box,
)


def test_boxes_overlap_ratio():
    a = (0, 0, 100, 100)
    b = (50, 50, 150, 150)
    assert boxes_overlap_ratio(a, b) > 0.2


def test_near_duplicate_box():
    a = (0, 0, 100, 100)
    b = (5, 5, 95, 95)
    assert is_near_duplicate_box(a, [b], threshold=0.55)


def test_figure_asset_priority_prefers_layout_over_fallback():
    box = (0, 0, 200, 200)
    assert figure_asset_priority("layout", box) > figure_asset_priority("layout_fallback", box)


def test_figure_asset_priority_prefers_larger_layout_crop():
    small = (0, 0, 100, 100)
    large = (0, 0, 300, 300)
    assert figure_asset_priority("layout", large) > figure_asset_priority("layout", small)

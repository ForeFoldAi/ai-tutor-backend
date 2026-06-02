"""Unit tests for layout-aware PDF figure extraction."""

from __future__ import annotations

from pathlib import Path

from app.services.image_service.pdf_layout_extraction import (
    BBox,
    detect_captions_from_blocks,
    is_page_background,
    pair_images_to_captions,
    pairing_distance,
    pairing_confidence_from_distance,
    LayoutCaption,
    LayoutImage,
    LayoutTextBlock,
)
from app.services.image_service.textbook_image_extraction import reject_figure_rect

BACKEND = Path(__file__).resolve().parents[1]
WEATHER_PDF = BACKEND / "uploads" / "CBSE" / "CLASS_9" / "Social" / "d57926c5_gees102.pdf"


def test_is_page_background_large_plate():
    page = BBox(0, 0, 595, 842)
    bg = BBox(-5, -39, 600, 850)
    assert is_page_background(bg, page) is True
    fig = BBox(100, 200, 400, 500)
    assert is_page_background(fig, page) is False


def test_reject_large_content_plate():
    """NCERT-style repeated ~46% page shell must be rejected."""
    page_w, page_h = 595.0, 842.0
    plate = BBox(56.1, 192.9, 522.4, 659.1)
    rejected, reason = reject_figure_rect(
        plate.x0, plate.y0, plate.x1, plate.y1, page_w, page_h
    )
    assert rejected is True
    assert reason == "large_content_plate"
    diagram = BBox(42.0, 211.2, 394.8, 396.1)
    rejected2, _ = reject_figure_rect(
        diagram.x0, diagram.y0, diagram.x1, diagram.y1, page_w, page_h
    )
    assert rejected2 is False


def test_pairing_prefers_figure_above_caption():
    page = BBox(0, 0, 595, 842)
    large_plate = BBox(56.1, 192.9, 522.4, 659.1)
    diagram = BBox(42.0, 211.2, 394.8, 396.1)
    cap = BBox(60, 380, 280, 400)
    assert pairing_distance(diagram, cap, page_bbox=page) < pairing_distance(
        large_plate, cap, page_bbox=page
    )


def test_pairing_distance_vertical_bias():
    img = BBox(100, 100, 300, 300)
    cap_below = BBox(120, 320, 280, 340)
    cap_far = BBox(500, 110, 580, 130)
    assert pairing_distance(img, cap_below) < pairing_distance(img, cap_far)


def test_spatial_pairing_not_index_order():
    images = [
        (LayoutImage(1, 0, BBox(50, 50, 200, 200), 1), b"a"),
        (LayoutImage(1, 1, BBox(50, 400, 200, 550), 2), b"b"),
    ]
    captions = [
        LayoutCaption(1, 0, "2.3.2", "Fig 2.3.2 Clouds", BBox(60, 560, 190, 580)),
        LayoutCaption(1, 1, "2.3.1", "Fig 2.3.1 Ants", BBox(60, 210, 190, 230)),
    ]
    page = BBox(0, 0, 595, 842)
    assignments, orphan_i, orphan_j = pair_images_to_captions(images, captions, page_bbox=page)
    assert not orphan_i and not orphan_j
    by_cap = {captions[j].figure_number: i for i, j, _ in assignments}
    assert by_cap["2.3.1"] == 0
    assert by_cap["2.3.2"] == 1


def test_detect_captions_minimal_label():
    blocks = [
        LayoutTextBlock(2, BBox(10, 500, 200, 520), "Fig. 2.2"),
        LayoutTextBlock(2, BBox(10, 400, 400, 480), "Weather is the day-to-day condition of the atmosphere."),
    ]
    caps = detect_captions_from_blocks(2, blocks)
    assert len(caps) == 1
    assert caps[0].figure_number == "2.2"
    assert "2.2" in caps[0].caption_text


def test_pairing_confidence_monotonic():
    assert pairing_confidence_from_distance(0) > pairing_confidence_from_distance(200)


def test_weather_chapter_multi_figure_pages():
    if not WEATHER_PDF.is_file():
        return
    from app.services.image_service.pdf_layout_extraction import extract_document_layout

    result = extract_document_layout(str(WEATHER_PDF), max_figures=96)
    by_page: dict[int, list[str]] = {}
    for fig in result.figures:
        if fig.figure_number:
            by_page.setdefault(fig.page_number, []).append(fig.figure_number)

    # Page 2 (1-based): Fig 2.2
    assert "2.2" in by_page.get(2, [])
    # Page 4: multiple 2.3.x figures
    p4 = by_page.get(4, [])
    assert len(p4) >= 2
    # Page 5: multiple 2.4.x
    p5 = by_page.get(5, [])
    assert len(p5) >= 3

    fig22 = next(f for f in result.figures if f.figure_number == "2.2")
    assert fig22.figure_context
    assert len(fig22.image_bytes) > 5000
    assert fig22.pairing_confidence > 0
    # Must be the weather diagram crop, not the ~46% page content plate
    assert fig22.image_bbox.area / (595 * 842) < 0.22

    fig29 = next(f for f in result.figures if f.figure_number == "2.9")
    assert fig29.image_bbox.area / (595 * 842) < 0.22

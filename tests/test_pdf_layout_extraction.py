"""Tests for ML-only PDF extraction entry point."""

from __future__ import annotations

from pathlib import Path

from app.services.image_service.pdf_extraction_types import (
    BBox,
    figure_number_to_filename,
    validation_report_dict,
)
from app.services.image_service.textbook_image_extraction import reject_figure_rect

BACKEND = Path(__file__).resolve().parents[1]
GEES102 = BACKEND / "gees102.pdf"


def test_figure_number_to_filename():
    assert figure_number_to_filename("2.3.1", 3, 0) == "fig_2_3_1.jpg"
    assert figure_number_to_filename(None, 2, 5) == "p2_5.jpg"


def test_reject_full_page_plate():
    page_w, page_h = 595.0, 842.0
    plate = BBox(0, 0, page_w, page_h)
    rejected, reason = reject_figure_rect(
        plate.x0, plate.y0, plate.x1, plate.y1, page_w, page_h
    )
    assert rejected is True
    assert reason == "area_gt_70pct"


def test_validation_report_dict_empty():
    from app.services.image_service.pdf_extraction_types import DocumentExtractionResult

    report = validation_report_dict(DocumentExtractionResult(figures=[], page_logs=[]))
    assert report["total_figures"] == 0


def test_ml_extract_gees102():
    if not GEES102.is_file():
        return
    from app.services.image_service.pdf_extraction_types import extract_document_layout

    result = extract_document_layout(str(GEES102), max_figures=96)
    assert len(result.figures) >= 10
    numbers = {f.figure_number for f in result.figures if f.figure_number}
    assert "2.1" in numbers
    assert "2.2" in numbers
    for fig in result.figures:
        assert len(fig.image_bytes) > 1000
        assert fig.source_type.startswith("ml_")

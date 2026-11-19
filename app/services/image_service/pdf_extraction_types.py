"""Shared types and helpers for ML PDF figure extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BBox:
    """Axis-aligned box in PDF page coordinates (points, origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


@dataclass
class PairedFigure:
    """One extracted figure/table asset ready for persistence."""

    page_number: int
    page_index: int
    sequence: int
    figure_number: str | None
    caption: str | None
    figure_context: str
    image_bbox: BBox
    caption_bbox: BBox | None
    pairing_distance: float
    pairing_confidence: float
    image_bytes: bytes
    nearby_before: str = ""
    nearby_after: str = ""
    section_title: str | None = None
    subsection_title: str | None = None
    validation_flags: list[str] = field(default_factory=list)
    source_type: str = "ml_layout"
    caption_source: str = "none"
    page_coverage: float = 0.0
    confidence: float = 1.0
    preferred_file_name: str | None = None


@dataclass
class PageExtractionLog:
    page_number: int
    backgrounds_removed: int = 0
    images_found: int = 0
    captions_found: int = 0
    pairs: list[dict[str, Any]] = field(default_factory=list)
    orphan_captions: list[str] = field(default_factory=list)
    orphan_images: list[int] = field(default_factory=list)
    suspicious_pairs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentExtractionResult:
    figures: list[PairedFigure]
    page_logs: list[PageExtractionLog]
    ml_assets: list[Any] = field(default_factory=list)


def bbox_to_json(bbox: BBox | None) -> str | None:
    if bbox is None:
        return None
    return json.dumps(bbox.to_list())


def bbox_from_json(raw: str | None) -> BBox | None:
    if not raw:
        return None
    try:
        vals = json.loads(raw)
        if isinstance(vals, list) and len(vals) == 4:
            return BBox(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def figure_number_to_filename(figure_number: str | None, page_index: int, seq: int) -> str:
    if figure_number:
        parts = re.sub(r"[^\d.]", "", figure_number.strip()).strip(".").split(".")
        parts = [p for p in parts if p]
        if parts:
            return f"fig_{'_'.join(parts)}.jpg"
    return f"p{page_index}_{seq}.jpg"


def validation_report_dict(result: DocumentExtractionResult) -> dict[str, Any]:
    return {
        "total_figures": len(result.figures),
        "pages": [
            {
                "page": pl.page_number,
                "backgrounds_removed": pl.backgrounds_removed,
                "images_found": pl.images_found,
                "captions_found": pl.captions_found,
                "pairs": pl.pairs,
                "orphan_captions": pl.orphan_captions,
                "orphan_images": pl.orphan_images,
                "suspicious_pairs": pl.suspicious_pairs,
            }
            for pl in result.page_logs
        ],
    }


def extract_document_layout(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    max_figures: int = 96,
    chapter_title: str | None = None,
) -> DocumentExtractionResult:
    """Extract figures via the native ML raster pipeline."""
    from app.services.pdf_extract_pipeline import pipeline_enabled
    from app.services.pdf_extract_pipeline.bridge import extract_with_ml_pipeline

    if not pipeline_enabled():
        raise RuntimeError(
            "PDF_EXTRACTION_PIPELINE_ENABLED must be true. "
            "Legacy extraction has been removed."
        )

    result, ml_assets, _pipeline = extract_with_ml_pipeline(
        pdf_path,
        max_pages=max_pages,
        max_figures=max_figures,
        chapter_title=chapter_title,
    )
    result.ml_assets = ml_assets
    return result

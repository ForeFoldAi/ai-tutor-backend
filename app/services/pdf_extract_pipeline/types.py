"""Shared data structures for the PDF extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


def poly_to_bbox(poly: list[float]) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = poly[0], poly[1], poly[4], poly[5]
    return int(xmin), int(ymin), int(xmax), int(ymax)


def bbox_to_poly(xmin: int, ymin: int, xmax: int, ymax: int) -> list[int]:
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


@dataclass
class LayoutElement:
    category_type: str
    poly: list[float]
    score: float = 0.0
    text: str = ""
    latex: str = ""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return poly_to_bbox(self.poly)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "category_type": self.category_type,
            "poly": self.poly,
            "score": self.score,
        }
        if self.text:
            out["text"] = self.text
        if self.latex:
            out["latex"] = self.latex
        return out


@dataclass
class PageInfo:
    page_no: int
    width: int
    height: int


@dataclass
class PageExtraction:
    page_no: int
    layout_dets: list[LayoutElement] = field(default_factory=list)
    page_info: PageInfo | None = None
    markdown: str = ""
    table_structured: dict[int, str] = field(default_factory=dict)
    pil_image: Image.Image | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_dets": [d.to_dict() for d in self.layout_dets],
            "page_info": {
                "page_no": self.page_info.page_no if self.page_info else self.page_no,
                "width": self.page_info.width if self.page_info else 0,
                "height": self.page_info.height if self.page_info else 0,
            },
            "markdown": self.markdown,
        }


@dataclass
class ExtractedAsset:
    """A cropped figure, table, or formula image plus metadata."""

    asset_type: str  # figure | table | formula
    page_no: int
    image_bytes: bytes
    number: str | None = None
    caption: str = ""
    structured_text: str = ""  # LaTeX/HTML for tables/formulas
    nearby_before: str = ""
    nearby_after: str = ""
    bbox: tuple[int, int, int, int] | None = None
    source: str = "layout"


@dataclass
class PersistableMlAsset:
    """Shape expected by textbook_image_extraction._persist_ml_asset_row."""

    content_kind: str
    page_index: int
    sequence: int
    image_bytes: bytes
    figure_number: str | None = None
    caption: str = ""
    structured_content: str = ""
    page_markdown: str = ""
    image_bbox: tuple[float, float, float, float] | None = None
    source_type: str = "ml_layout"

    @classmethod
    def from_extracted(
        cls,
        asset: ExtractedAsset,
        sequence: int,
        *,
        page_markdown: str = "",
    ) -> PersistableMlAsset:
        bbox = None
        if asset.bbox:
            bbox = (float(asset.bbox[0]), float(asset.bbox[1]), float(asset.bbox[2]), float(asset.bbox[3]))
        return cls(
            content_kind=asset.asset_type,
            page_index=max(0, asset.page_no - 1),
            sequence=sequence,
            image_bytes=asset.image_bytes,
            figure_number=asset.number,
            caption=asset.caption,
            structured_content=asset.structured_text,
            page_markdown=page_markdown,
            image_bbox=bbox,
            source_type=f"ml_{asset.source}",
        )


@dataclass
class PipelineResult:
    pages: list[PageExtraction] = field(default_factory=list)
    assets: list[ExtractedAsset] = field(default_factory=list)
    full_text: str = ""

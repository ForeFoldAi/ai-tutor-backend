"""
Multi-model PDF extraction orchestrator.

Per-page flow:
  1. Rasterize page at configured DPI
  2. Layout detection (DocLayout-YOLO)
  3. Formula detection (YOLO MFD) — optional
  4. Batch formula recognition (UniMERNet) — optional
  5. OCR on text regions with formula masking — optional
  6. Table VLM on table crops — optional
  7. Merge spans → page markdown
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import fitz
from PIL import Image

from app.services.image_service.formula_bbox import expand_formula_bbox
from app.services.pdf_extract_pipeline.config import (
    extraction_dpi,
    load_pipeline_config,
    stage_enabled,
)
from app.services.pdf_extract_pipeline.figure_pairing import extract_assets_from_page
from app.services.pdf_extract_pipeline.merge import latex_rm_whitespace, page_to_markdown
from app.services.pdf_extract_pipeline.models.formula_detection import FormulaDetector
from app.services.pdf_extract_pipeline.models.formula_recognition import FormulaRecognizer
from app.services.pdf_extract_pipeline.models.layout import LayoutDetector
from app.services.pdf_extract_pipeline.models.ocr import OcrEngine
from app.services.pdf_extract_pipeline.models.table_vlm import TableVlmParser
from app.services.pdf_extract_pipeline.raster import load_pdf_pages
from app.services.pdf_extract_pipeline.types import (
    LayoutElement,
    PageExtraction,
    PageInfo,
    PipelineResult,
)

logger = logging.getLogger(__name__)

_OCR_REGION_TYPES = {"title", "plain text", "figure_caption", "table_caption", "table_footnote", "formula_caption"}
_TABLE_TYPE = "table"
_MIN_TEXT_LAYER_CHARS = 20
_RUN_LOCK = threading.Lock()


def _crop_region(image: Image.Image, element: LayoutElement, padding: int = 25) -> tuple[Image.Image, list[int]]:
    xmin, ymin, xmax, ymax = element.bbox
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(image.width, xmax + padding)
    ymax = min(image.height, ymax + padding)
    crop = image.crop((xmin, ymin, xmax, ymax))
    return crop, [padding, padding, xmin, ymin, xmax, ymax, crop.width, crop.height]


class PdfExtractionPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_pipeline_config()
        self.dpi = int(self.config.get("dpi", extraction_dpi()))
        self._layout: LayoutDetector | None = None
        self._mfd: FormulaDetector | None = None
        self._mfr: FormulaRecognizer | None = None
        self._ocr: OcrEngine | None = None
        self._table_vlm: TableVlmParser | None = None

    @property
    def layout(self) -> LayoutDetector:
        if self._layout is None:
            self._layout = LayoutDetector(self.config.get("layout_detection", {}))
        return self._layout

    @property
    def mfd(self) -> FormulaDetector:
        if self._mfd is None:
            self._mfd = FormulaDetector(self.config.get("formula_detection", {}))
        return self._mfd

    @property
    def mfr(self) -> FormulaRecognizer:
        if self._mfr is None:
            self._mfr = FormulaRecognizer(self.config.get("formula_recognition", {}))
        return self._mfr

    @property
    def ocr(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = OcrEngine(self.config.get("ocr", {}))
        return self._ocr

    @property
    def table_vlm(self) -> TableVlmParser:
        if self._table_vlm is None:
            self._table_vlm = TableVlmParser(self.config.get("table_vlm", {}))
        return self._table_vlm

    def process_pdf(
        self,
        pdf_path: str,
        *,
        max_pages: int | None = None,
    ) -> PipelineResult:
        # ponytail: one chapter at a time per process bounds RAM; parallelism comes from more worker processes.
        with _RUN_LOCK:
            return self._process_pdf(pdf_path, max_pages=max_pages)

    def _process_pdf(self, pdf_path: str, *, max_pages: int | None) -> PipelineResult:
        t0 = time.time()
        doc, images = load_pdf_pages(pdf_path, dpi=self.dpi, max_pages=max_pages)
        n_pages = len(images)

        pages: list[PageExtraction] = []
        formula_images: list[Image.Image] = []
        formula_targets: list[LayoutElement] = []

        for page_idx, image in enumerate(images):
            w, h = image.size
            layout_dets = self.layout.predict(image)

            if stage_enabled(self.config, "formula_detection") and self.mfd.available():
                try:
                    mfd_dets = self.mfd.predict(image)
                    layout_dets.extend(mfd_dets)
                    if stage_enabled(self.config, "formula_recognition") and self.mfr.available():
                        for det in mfd_dets:
                            xmin, ymin, xmax, ymax = expand_formula_bbox(
                                det.bbox, image.size
                            )
                            formula_images.append(image.crop((xmin, ymin, xmax, ymax)))
                            formula_targets.append(det)
                except Exception as exc:
                    logger.warning("Formula detection failed page %d: %s", page_idx + 1, exc)

            pages.append(
                PageExtraction(
                    page_no=page_idx,
                    layout_dets=layout_dets,
                    page_info=PageInfo(page_no=page_idx, width=w, height=h),
                    pil_image=image,
                )
            )

        if formula_targets and stage_enabled(self.config, "formula_recognition"):
            try:
                latex_list = self.mfr.recognize_batch(formula_images)
                for det, latex in zip(formula_targets, latex_list):
                    det.latex = latex_rm_whitespace(latex)
            except Exception as exc:
                logger.warning("Formula recognition failed: %s", exc)

        for page_ex in pages:
            image = page_ex.pil_image
            if image is None:
                continue
            pdf_page = doc[page_ex.page_no]
            regions = [d for d in page_ex.layout_dets if d.category_type in _OCR_REGION_TYPES]
            if len((pdf_page.get_text() or "").strip()) >= _MIN_TEXT_LAYER_CHARS:
                # Digital page: the text layer is exact and free; OCR is only for scanned pages.
                scale = 72.0 / self.dpi
                for det in regions:
                    x0, y0, x1, y1 = det.bbox
                    text = " ".join(pdf_page.get_textbox(fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)).split())
                    if text:
                        page_ex.layout_dets.append(LayoutElement(category_type="text", poly=list(det.poly), score=1.0, text=text))
            elif stage_enabled(self.config, "ocr") and self.ocr.available():
                ocr_start = time.time()
                for det in regions:
                    crop, meta = _crop_region(image, det)
                    paste_x, paste_y, xmin, ymin, _, _, _, _ = meta
                    try:
                        ocr_elems = self.ocr.ocr_region(crop)
                        for elem in ocr_elems:
                            pts = elem.poly
                            mapped = []
                            for i in range(0, len(pts), 2):
                                mapped.extend([
                                    pts[i] - paste_x + xmin,
                                    pts[i + 1] - paste_y + ymin,
                                ])
                            elem.poly = mapped
                            page_ex.layout_dets.append(elem)
                    except Exception as exc:
                        logger.warning("OCR region failed page %d: %s", page_ex.page_no + 1, exc)
                logger.debug("Page %d OCR %.2fs", page_ex.page_no + 1, time.time() - ocr_start)

            if stage_enabled(self.config, "table_vlm") and self.table_vlm.available():
                table_crops: list[Image.Image] = []
                for det in page_ex.layout_dets:
                    if det.category_type != _TABLE_TYPE:
                        continue
                    crop, _ = _crop_region(image, det, padding=5)
                    table_crops.append(crop)
                if table_crops:
                    try:
                        parsed = self.table_vlm.parse_tables(table_crops)
                        page_ex.table_structured = {
                            j: text for j, text in enumerate(parsed) if text
                        }
                    except Exception as exc:
                        logger.warning("Table VLM failed page %d: %s", page_ex.page_no + 1, exc)

            page_ex.markdown = page_to_markdown([d.to_dict() for d in page_ex.layout_dets])

        document_has_fig_numbers = any(
            "fig." in (doc[i].get_text() or "").lower() for i in range(n_pages)
        )

        assets = []
        assigned_fig_numbers: set[str] = set()
        for page_idx in range(n_pages):
            page_ex = pages[page_idx]
            page_assigned_boxes: list[tuple[int, int, int, int]] = []
            page_assets = extract_assets_from_page(
                page_ex,
                doc[page_idx],
                dpi=self.dpi,
                document_has_fig_numbers=document_has_fig_numbers,
                table_structured=page_ex.table_structured,
                assigned_fig_numbers=assigned_fig_numbers,
                assigned_figure_boxes=page_assigned_boxes,
            )
            for asset in page_assets:
                if asset.asset_type == "figure" and asset.number:
                    assigned_fig_numbers.add(asset.number)
                if asset.asset_type == "figure" and asset.bbox:
                    page_assigned_boxes.append(asset.bbox)
            assets.extend(page_assets)

        full_text = "\n\n".join(p.markdown for p in pages if p.markdown)
        doc.close()
        for page_ex in pages:
            page_ex.pil_image = None  # assets are already cropped; rasters would pin RAM via the result cache
        logger.info(
            "[ML-PIPELINE] %s pages=%d assets=%d elapsed=%.1fs",
            pdf_path,
            n_pages,
            len(assets),
            time.time() - t0,
        )
        return PipelineResult(pages=pages, assets=assets, full_text=full_text)


_PIPELINE: PdfExtractionPipeline | None = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> PdfExtractionPipeline:
    """Process-wide pipeline so layout/formula/OCR models load once, not once per chapter."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = PdfExtractionPipeline()
        return _PIPELINE

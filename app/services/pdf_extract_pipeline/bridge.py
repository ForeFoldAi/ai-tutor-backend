"""
ML-only extraction bridge — converts pipeline assets to PairedFigure rows.

No caption-anchored fallback. If the ML pipeline fails, extraction fails.
"""

from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

from app.services.image_service.figure_filters import figure_asset_priority
from app.services.image_service.pdf_extraction_types import (
    BBox,
    DocumentExtractionResult,
    PageExtractionLog,
    PairedFigure,
)
from app.services.image_service.textbook_image_display import normalize_image_blob
from app.services.pdf_extract_pipeline.pipeline import PdfExtractionPipeline
from app.services.pdf_extract_pipeline.types import ExtractedAsset, PersistableMlAsset, PipelineResult

logger = logging.getLogger(__name__)


class MlPipelineExtractionError(RuntimeError):
    """Raised when the ML extraction pipeline cannot complete."""


def _asset_to_paired_figure(
    asset: ExtractedAsset,
    seq: int,
    chapter_title: str | None,
) -> PairedFigure | None:
    try:
        jpeg = normalize_image_blob(asset.image_bytes, preserve_figure_crop=True)
    except Exception:
        try:
            img = Image.open(BytesIO(asset.image_bytes)).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=88)
            jpeg = buf.getvalue()
        except Exception as exc:
            logger.debug("Skip asset page=%d: %s", asset.page_no, exc)
            return None

    if not jpeg:
        return None

    bbox = asset.bbox or (0, 0, 100, 100)
    image_bbox = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    page_index = max(0, asset.page_no - 1)
    caption = asset.caption or ""
    nearby_before = getattr(asset, "nearby_before", "") or ""
    nearby_after = getattr(asset, "nearby_after", "") or ""
    ctx_parts = [p for p in (chapter_title, nearby_before, caption, nearby_after, asset.structured_text) if p]
    figure_context = "\n".join(ctx_parts)[:4000]

    fname = None
    if asset.number and asset.asset_type == "figure":
        safe_num = asset.number.replace(".", "_")
        fname = f"fig_{safe_num}.jpg"

    return PairedFigure(
        page_number=asset.page_no,
        page_index=page_index,
        sequence=seq,
        figure_number=asset.number if asset.asset_type == "figure" else None,
        caption=caption or None,
        figure_context=figure_context,
        nearby_before=nearby_before,
        nearby_after=nearby_after,
        image_bbox=image_bbox,
        caption_bbox=None,
        pairing_distance=0.0,
        pairing_confidence=0.85 if asset.source == "layout" else 0.65,
        image_bytes=jpeg,
        source_type=f"ml_{asset.asset_type}",
        caption_source="detected" if caption else "none",
        confidence=0.85,
        preferred_file_name=fname,
    )


def _dedupe_figure_assets(assets: list[ExtractedAsset]) -> list[ExtractedAsset]:
    """Keep the best crop when multiple assets share a figure number."""
    numbered: dict[str, ExtractedAsset] = {}
    unnumbered: list[ExtractedAsset] = []
    for asset in assets:
        if asset.asset_type != "figure":
            continue
        if asset.number:
            current = numbered.get(asset.number)
            if current is None or figure_asset_priority(asset.source, asset.bbox) > figure_asset_priority(
                current.source, current.bbox
            ):
                numbered[asset.number] = asset
        else:
            unnumbered.append(asset)
    return list(numbered.values()) + unnumbered


def _build_page_logs(pipeline: PipelineResult) -> list[PageExtractionLog]:
    logs: list[PageExtractionLog] = []
    for page in pipeline.pages:
        figures = sum(1 for a in pipeline.assets if a.page_no == page.page_no + 1 and a.asset_type == "figure")
        tables = sum(1 for a in pipeline.assets if a.page_no == page.page_no + 1 and a.asset_type == "table")
        logs.append(
            PageExtractionLog(
                page_number=page.page_no + 1,
                images_found=figures,
                captions_found=figures,
                pairs=[{"tables": tables}],
            )
        )
    return logs


def extract_with_ml_pipeline(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    max_figures: int = 96,
    chapter_title: str | None = None,
) -> tuple[DocumentExtractionResult, list[PersistableMlAsset], PipelineResult]:
    """
    Run the native ML raster pipeline only. No caption-anchored merge.
    """
    from app.services.pdf_extract_pipeline.cache import (
        get_cached_pipeline_result,
        set_cached_pipeline_result,
    )

    ml_result = get_cached_pipeline_result(pdf_path)
    if ml_result is None:
        try:
            pipeline_runner = PdfExtractionPipeline()
            ml_result = pipeline_runner.process_pdf(pdf_path, max_pages=max_pages)
            set_cached_pipeline_result(pdf_path, ml_result)
        except Exception as exc:
            raise MlPipelineExtractionError(
                f"ML extraction pipeline failed for {pdf_path}: {exc}"
            ) from exc

    if not ml_result.pages:
        raise MlPipelineExtractionError(f"ML pipeline returned no pages for {pdf_path}")

    ml_figures: list[PairedFigure] = []
    figure_assets = _dedupe_figure_assets([a for a in ml_result.assets if a.asset_type == "figure"])
    for seq, asset in enumerate(figure_assets):
        if len(ml_figures) >= max_figures:
            break
        pf = _asset_to_paired_figure(asset, seq=seq, chapter_title=chapter_title)
        if pf:
            ml_figures.append(pf)

    page_markdown_by_no = {page.page_no: page.markdown for page in ml_result.pages}
    persistable: list[PersistableMlAsset] = []
    for seq, asset in enumerate(ml_result.assets):
        if asset.asset_type in ("table", "formula"):
            persistable.append(
                PersistableMlAsset.from_extracted(
                    asset,
                    sequence=1000 + seq,
                    page_markdown=page_markdown_by_no.get(asset.page_no, ""),
                )
            )

    logger.info(
        "[ML-ONLY] figures=%d tables=%d formulas=%d pages=%d",
        len(ml_figures),
        sum(1 for a in ml_result.assets if a.asset_type == "table"),
        sum(1 for a in ml_result.assets if a.asset_type == "formula"),
        len(ml_result.pages),
    )

    return (
        DocumentExtractionResult(
            figures=ml_figures,
            page_logs=_build_page_logs(ml_result),
            ml_assets=persistable,
        ),
        persistable,
        ml_result,
    )

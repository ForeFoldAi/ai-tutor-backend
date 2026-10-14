"""PDF extraction adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def page_count(pdf_path: Path) -> int:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return int(n)
    except Exception:
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(pdf_path)).pages)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Cannot read page count for {pdf_path}: {exc}") from exc


def extract_pdf(pdf_path: Path, *, use_cache: bool = True) -> Any:
    """Run the production PdfExtractionPipeline and return PipelineResult."""
    from app.services.pdf_extract_pipeline.cache import (
        get_cached_pipeline_result,
        set_cached_pipeline_result,
    )
    from app.services.pdf_extract_pipeline.pipeline import PdfExtractionPipeline

    path = str(pdf_path.resolve())
    if use_cache:
        cached = get_cached_pipeline_result(path)
        if cached is not None:
            return cached
    result = PdfExtractionPipeline().process_pdf(path)
    if use_cache:
        set_cached_pipeline_result(path, result)
    return result


def pipeline_to_serializable(result: Any) -> dict[str, Any]:
    pages = []
    for p in getattr(result, "pages", []) or []:
        pages.append(
            {
                "page_no": p.page_no,
                "markdown": (p.markdown or "")[:4000],
                "layout_count": len(p.layout_dets or []),
                "table_count": len(p.table_structured or {}),
                "width": p.page_info.width if p.page_info else None,
                "height": p.page_info.height if p.page_info else None,
            }
        )
    assets = []
    for a in getattr(result, "assets", []) or []:
        assets.append(
            {
                "asset_type": a.asset_type,
                "page_no": a.page_no,
                "number": a.number,
                "caption": a.caption,
                "bbox": a.bbox,
                "bytes_len": len(a.image_bytes or b""),
                "structured_text_preview": (a.structured_text or "")[:500],
            }
        )
    return {
        "page_count": len(pages),
        "pages": pages,
        "assets": assets,
        "full_text_len": len(getattr(result, "full_text", "") or ""),
    }

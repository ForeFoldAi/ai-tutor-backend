"""
OCR service for scanned textbook PDFs.

Strategy:
  1. Quality probe — check if PyMuPDF text layer has sufficient density.
     PDFs with < MIN_CHARS_PER_PAGE average text are treated as scanned.
  2. If sparse, attempt Tesseract OCR via pytesseract (if installed).
  3. As a secondary fallback, try PyMuPDF's built-in OCR (via Tesseract C-lib).
  4. Store extracted OCR text on TextbookImage.ocr_text and mark
     TextbookUpload.ocr_status = EMBEDDED.

This module is OPTIONAL — if neither Tesseract nor PyMuPDF OCR is available
the system continues with text-layer-only mode and logs a warning.

Configuration (from app/config.py):
  OCR_ENABLED       — master on/off switch (default true)
  OCR_LANG          — Tesseract language code(s) (default "eng")
  OCR_MIN_CHARS_PER_PAGE — threshold below which a page is "scanned" (default 80)
  OCR_DPI           — render DPI for scanned page rasterisation (default 200)
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality probe constants
# ---------------------------------------------------------------------------

_MIN_CHARS_PER_PAGE_DEFAULT = 80
_OCR_DPI_DEFAULT = 200
_OCR_LANG_DEFAULT = "eng"


def _ocr_enabled() -> bool:
    from app.config import OCR_ENABLED
    return OCR_ENABLED


def _ocr_lang() -> str:
    from app.config import OCR_LANG
    return OCR_LANG or _OCR_LANG_DEFAULT


def _ocr_dpi() -> int:
    from app.config import OCR_DPI
    return OCR_DPI or _OCR_DPI_DEFAULT


def _min_chars() -> int:
    from app.config import OCR_MIN_CHARS_PER_PAGE
    return OCR_MIN_CHARS_PER_PAGE or _MIN_CHARS_PER_PAGE_DEFAULT


# ---------------------------------------------------------------------------
# Text-layer quality probe
# ---------------------------------------------------------------------------

def pdf_is_likely_scanned(pdf_path: str, *, sample_pages: int = 8) -> bool:
    """
    Return True if the PDF text layer is too sparse to be a native PDF.

    Samples up to *sample_pages* evenly distributed pages; returns True
    when the average character count per sampled page is below threshold.
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        n = len(doc)
        step = max(1, n // sample_pages)
        indices = list(range(0, n, step))[:sample_pages]
        total_chars = 0
        sampled = 0
        for i in indices:
            try:
                text = doc[i].get_text("text") or ""
                total_chars += len(text.strip())
                sampled += 1
            except Exception:
                continue
        doc.close()
        if sampled == 0:
            return True
        avg = total_chars / sampled
        is_scanned = avg < _min_chars()
        if is_scanned:
            logger.info(
                "[OCR] PDF likely scanned (avg %.0f chars/page < %d): %s",
                avg, _min_chars(), os.path.basename(pdf_path),
            )
        return is_scanned
    except Exception as exc:
        logger.debug("pdf_is_likely_scanned probe failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Tesseract OCR via pytesseract
# ---------------------------------------------------------------------------

def _try_pytesseract(image_bytes: bytes, lang: str) -> str | None:
    """Run pytesseract on image bytes; returns text or None if unavailable."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip() or None
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("pytesseract OCR failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# PyMuPDF built-in OCR (requires Tesseract C library, not the Python binding)
# ---------------------------------------------------------------------------

def _try_pymupdf_ocr(pdf_path: str, page_index: int, dpi: int, lang: str) -> str | None:
    """
    Use PyMuPDF's page.get_textpage_ocr() when Tesseract is installed as a system lib.
    Falls back gracefully if OCR support is not compiled in.
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            page = doc[page_index]
            # get_textpage_ocr is only available in PyMuPDF >= 1.19 with OCR support
            tp = page.get_textpage_ocr(dpi=dpi, full=True, language=lang)
            text = page.get_text(textpage=tp).strip()
            return text or None
        except AttributeError:
            return None
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("PyMuPDF OCR fallback failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-image OCR (for caption region extraction)
# ---------------------------------------------------------------------------

def ocr_image_bytes(image_bytes: bytes, *, lang: str | None = None) -> str | None:
    """
    Run OCR on raw image bytes; returns cleaned text or None.

    Used to extract text from embedded figure images when the PDF text layer
    is sparse (scanned textbooks) or when the caption is embedded inside the
    image raster itself.
    """
    if not _ocr_enabled():
        return None
    effective_lang = lang or _ocr_lang()
    return _try_pytesseract(image_bytes, effective_lang)


# ---------------------------------------------------------------------------
# Whole-page OCR (for scanned PDFs)
# ---------------------------------------------------------------------------

def ocr_page(pdf_path: str, page_index: int) -> str | None:
    """
    OCR a single PDF page; returns plain text or None.

    Tries:
      1. pytesseract on a rasterised page image
      2. PyMuPDF built-in OCR
    """
    if not _ocr_enabled():
        return None

    lang = _ocr_lang()
    dpi = _ocr_dpi()

    # Rasterise page → JPEG bytes for pytesseract
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            page = doc[page_index]
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
        finally:
            doc.close()
        result = _try_pytesseract(img_bytes, lang)
        if result:
            return result
    except Exception as exc:
        logger.debug("Page rasterise for OCR failed: %s", exc)

    # PyMuPDF built-in OCR fallback
    return _try_pymupdf_ocr(pdf_path, page_index, dpi, lang)


# ---------------------------------------------------------------------------
# Upload-level OCR orchestration
# ---------------------------------------------------------------------------

def ocr_upload_if_needed(
    db: Any,
    upload: Any,
    images: list[Any],
    *,
    force: bool = False,
) -> int:
    """
    For scanned uploads, run OCR on each page and populate TextbookImage.ocr_text.

    Skips uploads where ocr_status is already EMBEDDED (unless force=True).
    Returns the number of images that received OCR text.
    """
    from app.modules.catalog.models import ProcessingStatusEnum

    if not _ocr_enabled():
        return 0

    status = getattr(upload, "ocr_status", None)
    if not force and status == ProcessingStatusEnum.EMBEDDED:
        return 0

    pdf_path = getattr(upload, "file_path", None) or ""
    if not pdf_path or not os.path.isfile(pdf_path):
        return 0

    ext = os.path.splitext(pdf_path)[1].lower()
    if ext not in (".pdf",):
        # DOCX doesn't need page OCR (text is in XML)
        return 0

    if not pdf_is_likely_scanned(pdf_path):
        upload.ocr_status = ProcessingStatusEnum.EMBEDDED
        db.commit()
        return 0

    logger.info("[OCR] Starting page OCR for upload %s", upload.id)
    upload.ocr_status = ProcessingStatusEnum.PROCESSING
    db.commit()

    indexed = 0
    for im in images:
        if getattr(im, "ocr_text", None):
            continue
        page_idx = getattr(im, "page_index", 0)
        try:
            text = ocr_page(pdf_path, page_idx)
            if text:
                im.ocr_text = text[:4000]
                indexed += 1
        except Exception as exc:
            logger.warning("[OCR] page %d failed for %s: %s", page_idx, upload.id, exc)

    upload.ocr_status = ProcessingStatusEnum.EMBEDDED
    try:
        db.commit()
    except Exception as exc:
        logger.warning("[OCR] commit failed: %s", exc)
        db.rollback()

    logger.info("[OCR] Completed: %d/%d images received OCR text for %s", indexed, len(images), upload.id)
    return indexed

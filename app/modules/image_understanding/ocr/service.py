"""Cached, non-blocking OCR entry point used by image_understanding.service."""

from __future__ import annotations

import asyncio

from app.config import TESSERACT_ENABLED, TESSERACT_MAX_CONCURRENCY, TESSERACT_TIMEOUT_SEC
from app.modules.image_understanding.cache import get_cached_ocr, set_cached_ocr
from app.modules.image_understanding.ocr.schemas import OCRResult
from app.modules.image_understanding.ocr.tesseract import TesseractOCRService

_service = TesseractOCRService()
# Bounds tesseract processes and keeps OCR from starving the shared to_thread pool (RAG uses it too).
_slots = asyncio.Semaphore(TESSERACT_MAX_CONCURRENCY)
OCR_BUSY = "ocr_busy"


def ocr_engine_id() -> str:
    return f"tesseract-{_service.version() or 'unknown'}"


async def run_ocr(data: bytes, content_hash: str) -> tuple[OCRResult | None, bool]:
    """Returns (result, cache_hit). None when OCR is disabled. Errors come back as result.error."""
    if not TESSERACT_ENABLED:
        return None, False
    engine = ocr_engine_id()
    cached = get_cached_ocr(content_hash, engine, _service.lang)
    if cached is not None:
        return cached, True
    try:
        await asyncio.wait_for(_slots.acquire(), timeout=TESSERACT_TIMEOUT_SEC)
    except TimeoutError:
        return OCRResult(error=OCR_BUSY, language=_service.lang, engine=engine), False
    try:
        result = await asyncio.to_thread(_service.run, data)
    finally:
        _slots.release()
    result.engine = engine
    if not result.error:
        set_cached_ocr(content_hash, engine, _service.lang, result)
    return result, False


def ocr_status() -> dict:
    if not TESSERACT_ENABLED:
        return {"enabled": False, "status": "disabled"}
    st = _service.status()
    healthy = st["available"] and st["lang_available"]
    return {"enabled": True, **st, "status": "ok" if healthy else "degraded", "max_concurrency": TESSERACT_MAX_CONCURRENCY}

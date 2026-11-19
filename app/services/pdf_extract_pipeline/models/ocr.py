"""PaddleOCR 3.x for page regions without a usable PDF text layer."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
from PIL import Image

from app.services.pdf_extract_pipeline.types import LayoutElement

logger = logging.getLogger(__name__)


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


_ENGINES: dict[str, Any] = {}
_ENGINE_LOCK = threading.Lock()


def _paddle_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """PaddleOCR 3.x arguments. Model dirs that don't exist are dropped so the default models load."""
    kwargs: dict[str, Any] = {
        "lang": config.get("lang", "en"),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": bool(config.get("use_angle_cls", False)),
    }
    for old, new in (("det_model_dir", "text_detection_model_dir"), ("rec_model_dir", "text_recognition_model_dir")):
        path = config.get(old)
        if path and os.path.isdir(path):
            kwargs[new] = path
    return kwargs


def _parse_result(raw: Any) -> list[tuple[list[list[float]], str, float]]:
    """(points, text, score) per detected line from a PaddleOCR 3.x predict() result."""
    out: list[tuple[list[list[float]], str, float]] = []
    for res in raw or []:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        polys = res.get("rec_polys")
        if polys is None:
            polys = res.get("dt_polys") or []
        for poly, text, score in zip(polys, texts, scores):
            out.append(([[float(x), float(y)] for x, y in np.asarray(poly).tolist()], str(text), float(score)))
    return out


class OcrEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return self._ensure_ocr() is not None

    def _ensure_ocr(self) -> Any:
        """One engine per process; a failed build is remembered (None) so it is never retried per region."""
        key = repr(sorted(_paddle_kwargs(self.config).items()))
        if key in _ENGINES:
            return _ENGINES[key]
        with _ENGINE_LOCK:
            if key not in _ENGINES:
                try:
                    from paddleocr import PaddleOCR

                    _ENGINES[key] = PaddleOCR(**_paddle_kwargs(self.config))
                    logger.info("Loaded PaddleOCR (lang=%s)", self.config.get("lang", "en"))
                except Exception as exc:
                    _ENGINES[key] = None
                    logger.warning("PaddleOCR unavailable, OCR disabled for this process: %s", exc)
        return _ENGINES[key]

    def ocr_region(self, image: Image.Image, *, padding: int = 25) -> list[LayoutElement]:
        ocr = self._ensure_ocr() if self.available() else None
        if ocr is None:
            return []
        crop = image.copy()
        if padding:
            w, h = crop.size
            padded = Image.new("RGB", (w + padding * 2, h + padding * 2), "white")
            padded.paste(crop, (padding, padding))
            crop = padded

        elements: list[LayoutElement] = []
        for points, text, score in _parse_result(ocr.predict(_pil_to_cv2(crop))):
            flat = [c for p in points for c in (p[0] - padding, p[1] - padding)]
            elements.append(LayoutElement(category_type="text", poly=flat, score=round(score, 2), text=text))
        return elements

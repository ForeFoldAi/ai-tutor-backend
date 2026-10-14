"""PaddleOCR with formula-region masking."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.services.pdf_extract_pipeline.types import LayoutElement, bbox_to_poly

logger = logging.getLogger(__name__)


def _overlap_y(bbox1: list[float], bbox2: list[float], threshold: float = 0.8) -> bool:
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2
    overlap = max(0.0, min(y1_1, y1_2) - max(y0_1, y0_2))
    h1, h2 = y1_1 - y0_1, y1_2 - y0_2
    if min(h1, h2) <= 0:
        return False
    return (overlap / min(h1, h2)) > threshold


def _points_to_bbox(points: list[list[float]]) -> list[float]:
    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return [x0, y0, x1, y1]


def _bbox_to_points(bbox: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")


def _merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _remove_intervals(original: list[int], masks: list[list[int]]) -> list[list[int]]:
    merged_masks = _merge_intervals(masks)
    result: list[list[int]] = []
    start, end = original
    for ms, me in merged_masks:
        if ms > end:
            continue
        if me < start:
            continue
        if start < ms:
            result.append([start, ms - 1])
        start = max(me + 1, start)
    if start <= end:
        result.append([start, end])
    return result


def _update_det_boxes(dt_boxes: np.ndarray, mfd_res: list[dict[str, Any]]) -> list[np.ndarray]:
    new_boxes: list[np.ndarray] = []
    for text_box in dt_boxes:
        text_bbox = _points_to_bbox(text_box.tolist())
        masks: list[list[int]] = []
        for mf in mfd_res:
            mf_bbox = mf["bbox"]
            if _overlap_y(text_bbox, mf_bbox):
                masks.append([int(mf_bbox[0]), int(mf_bbox[2])])
        text_x = [int(text_bbox[0]), int(text_bbox[2])]
        for segment in _remove_intervals(text_x, masks):
            new_boxes.append(
                _bbox_to_points([segment[0], text_bbox[1], segment[1], text_bbox[3]])
            )
    return new_boxes


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class OcrEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._ocr: Any = None

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        from paddleocr import PaddleOCR

        kwargs = {k: v for k, v in self.config.items() if k != "show_log"}
        self._ocr = PaddleOCR(**kwargs)
        logger.info("Loaded PaddleOCR (lang=%s)", self.config.get("lang", "en"))
        return self._ocr

    def ocr_region(
        self,
        image: Image.Image,
        *,
        mfd_res: list[dict[str, Any]] | None = None,
        padding: int = 25,
    ) -> list[LayoutElement]:
        if not self.available():
            return []
        ocr = self._ensure_ocr()
        crop = image.copy()
        if padding:
            w, h = crop.size
            padded = Image.new("RGB", (w + padding * 2, h + padding * 2), "white")
            padded.paste(crop, (padding, padding))
            crop = padded

        cv_img = _pil_to_cv2(crop)
        raw = ocr.ocr(cv_img, cls=self.config.get("use_angle_cls", False))
        if not raw or not raw[0]:
            return []

        det_boxes = [np.array(box[0], dtype="float32") for box in raw[0]]
        if mfd_res:
            adjusted = []
            for mf in mfd_res:
                x0, y0, x1, y1 = mf["bbox"]
                adjusted.append({"bbox": [x0 + padding, y0 + padding, x1 + padding, y1 + padding]})
            det_boxes = _update_det_boxes(np.array(det_boxes), adjusted)

        elements: list[LayoutElement] = []
        for box_ocr in raw[0]:
            points, (text, score) = box_ocr
            pts = [[p[0] - padding, p[1] - padding] for p in points]
            flat = [c for p in pts for c in p]
            elements.append(
                LayoutElement(
                    category_type="text",
                    poly=flat,
                    score=round(float(score), 2),
                    text=str(text),
                )
            )
        return elements

"""YOLOv8 formula detection (inline + isolated)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.services.pdf_extract_pipeline.types import LayoutElement, bbox_to_poly

logger = logging.getLogger(__name__)

MFD_ID_TO_NAMES = {0: "inline", 1: "isolated"}


class FormulaDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.id_to_names = MFD_ID_TO_NAMES
        self.img_size = int(config.get("img_size", 1280))
        self.conf_thres = float(config.get("conf_thres", 0.25))
        self.iou_thres = float(config.get("iou_thres", 0.45))
        self._model: Any = None

    def available(self) -> bool:
        path = self.config.get("model_path", "")
        return bool(path and Path(path).is_file())

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        model_path = self.config.get("model_path", "")
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(f"Formula detection weights not found: {model_path}")
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        logger.info("Loaded formula detection YOLO from %s", model_path)
        return self._model

    def predict(self, image: Image.Image) -> list[LayoutElement]:
        if not self.available():
            return []
        model = self._ensure_model()
        results = model.predict(
            image,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        elements: list[LayoutElement] = []
        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            for xyxy, conf, cls in zip(
                result.boxes.xyxy.cpu(),
                result.boxes.conf.cpu(),
                result.boxes.cls.cpu(),
            ):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                cls_id = int(cls.item())
                elements.append(
                    LayoutElement(
                        category_type=self.id_to_names.get(cls_id, "inline"),
                        poly=bbox_to_poly(xmin, ymin, xmax, ymax),
                        score=round(float(conf.item()), 2),
                        latex="",
                    )
                )
        return elements

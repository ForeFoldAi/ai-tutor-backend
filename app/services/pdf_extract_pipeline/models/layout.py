"""DocLayout-YOLO layout detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.services.pdf_extract_pipeline.types import LayoutElement, bbox_to_poly

logger = logging.getLogger(__name__)

LAYOUT_ID_TO_NAMES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


class LayoutDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.id_to_names = LAYOUT_ID_TO_NAMES
        self.img_size = int(config.get("img_size", 1024))
        self.conf_thres = float(config.get("conf_thres", 0.25))
        self.iou_thres = float(config.get("iou_thres", 0.45))
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        model_path = self.config.get("model_path", "")
        hf_pretrained = self.config.get("hf_pretrained", "juliozhao/DocLayout-YOLO-DocStructBench")
        try:
            from doclayout_yolo import YOLOv10  # type: ignore[import]

            if model_path and Path(model_path).is_file():
                self._model = YOLOv10(model_path)
                logger.info("Loaded DocLayout-YOLO from %s", model_path)
            else:
                self._model = YOLOv10.from_pretrained(hf_pretrained)
                logger.info("Loaded DocLayout-YOLO from HuggingFace %s", hf_pretrained)
        except Exception as exc:
            logger.warning("DocLayout-YOLO unavailable (%s); trying ultralytics YOLO", exc)
            from ultralytics import YOLO

            if model_path and Path(model_path).is_file():
                self._model = YOLO(model_path)
            else:
                raise RuntimeError(
                    "Layout model unavailable: install doclayout-yolo or provide models/Layout/YOLO/doclayout_yolo_ft.pt"
                ) from exc
        return self._model

    def predict(self, image: Image.Image) -> list[LayoutElement]:
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
            names = getattr(result, "names", self.id_to_names)
            for xyxy, conf, cls in zip(
                result.boxes.xyxy.cpu(),
                result.boxes.conf.cpu(),
                result.boxes.cls.cpu(),
            ):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                cls_id = int(cls.item())
                label = names.get(cls_id, self.id_to_names.get(cls_id, "plain text"))
                elements.append(
                    LayoutElement(
                        category_type=label,
                        poly=bbox_to_poly(xmin, ymin, xmax, ymax),
                        score=round(float(conf.item()), 2),
                    )
                )
        return elements

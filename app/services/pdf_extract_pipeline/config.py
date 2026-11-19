"""Pipeline configuration loader."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(PROJECT_ROOT) / "configs" / "pdf_extraction_pipeline.yaml"


def pipeline_enabled() -> bool:
    from app.config import PDF_EXTRACTION_PIPELINE_ENABLED

    return PDF_EXTRACTION_PIPELINE_ENABLED


def models_root() -> Path:
    from app.config import PDF_EXTRACTION_MODELS_DIR

    return Path(PDF_EXTRACTION_MODELS_DIR)


def config_path() -> Path:
    from app.config import PDF_EXTRACTION_CONFIG_PATH

    return Path(PDF_EXTRACTION_CONFIG_PATH)


def extraction_dpi() -> int:
    from app.config import PDF_EXTRACTION_DPI

    return PDF_EXTRACTION_DPI


def table_vlm_enabled() -> bool:
    from app.config import PDF_EXTRACTION_ENABLE_TABLE_VLM

    return PDF_EXTRACTION_ENABLE_TABLE_VLM


def load_pipeline_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        logger.warning("Pipeline config missing at %s — using built-in defaults", path)
        return _default_config()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_paths(data)


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    root = models_root()

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, str) and node.startswith("models/"):
            # models_root already points at the models/ directory
            return str(root / node.removeprefix("models/"))
        return node

    return _walk(cfg)


def _default_config() -> dict[str, Any]:
    root = models_root()
    return {
        "dpi": extraction_dpi(),
        "stages": {
            "layout": True,
            "formula_detection": True,
            "formula_recognition": True,
            "ocr": True,
            "table_vlm": table_vlm_enabled(),
        },
        "layout_detection": {
            "img_size": 1024,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "model_path": str(root / "Layout" / "YOLO" / "doclayout_yolo_ft.pt"),
            "hf_pretrained": "juliozhao/DocLayout-YOLO-DocStructBench",
        },
        "formula_detection": {
            "img_size": 1280,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "model_path": str(root / "MFD" / "YOLO" / "yolo_v8_ft.pt"),
        },
        "formula_recognition": {
            "batch_size": 32,
            "model_path": str(root / "MFR" / "unimernet_tiny"),
            "cfg_path": str(Path(PROJECT_ROOT) / "configs" / "unimernet.yaml"),
        },
        "ocr": {
            "lang": "en",
            "use_angle_cls": False,
            "show_log": False,
            "det_model_dir": str(root / "OCR" / "PaddleOCR" / "det" / "en_PP-OCRv4_det"),
            "rec_model_dir": str(root / "OCR" / "PaddleOCR" / "rec" / "en_PP-OCRv4_rec"),
        },
        "table_vlm": {
            "model_path": "U4R/StructTable-InternVL2-1B",
            "output_format": "markdown",
            "batch_size": 1,
        },
    }


def stage_enabled(cfg: dict[str, Any], stage: str) -> bool:
    stages = cfg.get("stages") or {}
    if stage == "table_vlm":
        return bool(stages.get("table_vlm", False)) and table_vlm_enabled()
    return bool(stages.get(stage, True))

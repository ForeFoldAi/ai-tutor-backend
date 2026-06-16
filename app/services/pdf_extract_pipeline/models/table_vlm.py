"""StructEqTable vision-language table parsing."""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class TableVlmParser:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output_format = config.get("output_format", "markdown")
        self._model: Any = None

    def available(self) -> bool:
        try:
            import struct_eqtable  # noqa: F401

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        from struct_eqtable import build_model  # type: ignore[import]

        self._model = build_model(
            model_ckpt=self.config.get("model_path", "U4R/StructTable-InternVL2-1B"),
            max_new_tokens=int(self.config.get("max_new_tokens", 1024)),
            max_time=int(self.config.get("max_time", 30)),
            lmdeploy=bool(self.config.get("lmdeploy", False)),
            flash_attn=bool(self.config.get("flash_attn", True)),
            batch_size=int(self.config.get("batch_size", 1)),
        ).cuda()
        logger.info("Loaded StructEqTable VLM")
        return self._model

    def parse_tables(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        if not self.available():
            logger.warning("Table VLM unavailable (needs struct-eqtable + CUDA)")
            return [""] * len(images)
        model = self._ensure_model()
        fmt = self.output_format
        if fmt not in ("latex", "markdown", "html"):
            fmt = "markdown"
        return list(model(images, output_format=fmt))

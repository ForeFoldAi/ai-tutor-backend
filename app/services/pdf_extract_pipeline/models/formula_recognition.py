"""UniMERNet formula recognition → LaTeX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.services.pdf_extract_pipeline.merge import latex_rm_whitespace

logger = logging.getLogger(__name__)


class _FormulaImageDataset(Dataset):
    def __init__(self, images: list[Image.Image], transform: Any) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Any:
        img = self.images[idx].convert("RGB")
        return self.transform(img)


class FormulaRecognizer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.batch_size = int(config.get("batch_size", 32))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._vis_processor: Any = None

    def available(self) -> bool:
        model_dir = self.config.get("model_path", "")
        return bool(model_dir and Path(model_dir).is_dir())

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._vis_processor is not None:
            return self._model, self._vis_processor
        model_dir = self.config.get("model_path", "")
        cfg_path = self.config.get("cfg_path", "")
        if not model_dir or not Path(model_dir).is_dir():
            raise FileNotFoundError(f"UniMERNet weights not found: {model_dir}")
        import unimernet.tasks as tasks  # type: ignore[import]
        from unimernet.common.config import Config  # type: ignore[import]
        from unimernet.processors import load_processor  # type: ignore[import]

        args = argparse.Namespace(cfg_path=cfg_path, options=None)
        cfg = Config(args)
        cfg.config.model.pretrained = str(Path(model_dir) / "pytorch_model.pth")
        cfg.config.model.model_config.model_name = model_dir
        cfg.config.model.tokenizer_config.path = model_dir
        task = tasks.setup_task(cfg)
        self._model = task.build_model(cfg).to(self.device)
        self._vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )
        logger.info("Loaded UniMERNet from %s", model_dir)
        return self._model, self._vis_processor

    def recognize_batch(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        if not self.available():
            return [""] * len(images)
        model, vis_processor = self._ensure_model()
        dataset = _FormulaImageDataset(images, vis_processor)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0)
        results: list[str] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = model.generate({"image": batch})
                for pred in output["pred_str"]:
                    results.append(latex_rm_whitespace(pred))
        return results

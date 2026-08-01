"""Golden (expected) dataset loader and writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GoldenStore:
    """Loads goldens/<dataset>/<kind>.json and optional media folders."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def dataset_dir(self, dataset: str) -> Path:
        return self.root / dataset

    def load(self, dataset: str, kind: str, default: Any = None) -> Any:
        path = self.dataset_dir(dataset) / f"{kind}.json"
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, dataset: str, kind: str, data: Any) -> Path:
        d = self.dataset_dir(dataset)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{kind}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def exists(self, dataset: str, kind: str) -> bool:
        return (self.dataset_dir(dataset) / f"{kind}.json").exists()

    def media_dir(self, dataset: str, kind: str) -> Path:
        p = self.dataset_dir(dataset) / kind
        p.mkdir(parents=True, exist_ok=True)
        return p

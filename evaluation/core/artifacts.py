"""Artifact persistence under evaluation/artifacts/<run_id>/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: Path, run_id: str):
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, relative: str, data: Any) -> str:
        path = self.path(relative)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(path)

    def write_text(self, relative: str, text: str) -> str:
        path = self.path(relative)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def write_bytes(self, relative: str, data: bytes) -> str:
        path = self.path(relative)
        path.write_bytes(data)
        return str(path)

    def copy_file(self, src: Path, relative: str) -> str:
        dest = self.path(relative)
        shutil.copy2(src, dest)
        return str(dest)

    def dataset_dir(self, dataset: str, feature: str) -> Path:
        p = self.root / dataset / feature
        p.mkdir(parents=True, exist_ok=True)
        return p

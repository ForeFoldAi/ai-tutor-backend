"""Per-run evaluation context shared across phases."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.core.artifacts import ArtifactStore
from evaluation.core.config import EvalConfig
from evaluation.core.golden import GoldenStore


@dataclass
class DatasetBundle:
    """One PDF under evaluation plus resolved paths and identity."""

    name: str
    pdf_path: Path
    golden_dir: Path
    board: str
    class_level: str
    subject_name: str
    collection_name: str
    textbook_upload_id: str

    @property
    def stem(self) -> str:
        return self.pdf_path.stem


@dataclass
class EvalContext:
    config: EvalConfig
    run_id: str
    artifacts: ArtifactStore
    goldens: GoldenStore
    datasets: list[DatasetBundle] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def dataset_by_name(self, name: str) -> DatasetBundle:
        for d in self.datasets:
            if d.name == name or d.stem == name:
                return d
        raise KeyError(f"Dataset not found: {name}")

    def get_cached(self, key: str, factory):
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]

    def skip_llm(self) -> bool:
        return bool(self.config.get("phases", "skip_llm", default=False))

    def skip_heavy_ml(self) -> bool:
        return bool(self.config.get("phases", "skip_heavy_ml", default=False))


def build_context(config: EvalConfig | None = None, run_id: str | None = None) -> EvalContext:
    cfg = config or EvalConfig.load()
    rid = run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    artifacts = ArtifactStore(Path(cfg.get("paths", "artifacts")), rid)
    goldens = GoldenStore(Path(cfg.get("paths", "goldens")))
    board = cfg.get("datasets", "board", default="CBSE")
    class_level = cfg.get("datasets", "class_level", default="CLASS_9")
    subject = cfg.get("datasets", "subject_name", default="Science")
    prefix = cfg.get("datasets", "collection_prefix", default="eval")
    datasets: list[DatasetBundle] = []
    for pdf in cfg.pdf_paths():
        if not pdf.exists():
            continue
        stem = pdf.stem
        datasets.append(
            DatasetBundle(
                name=stem,
                pdf_path=pdf.resolve(),
                golden_dir=Path(cfg.get("paths", "goldens")) / stem,
                board=board,
                class_level=class_level,
                subject_name=subject,
                collection_name=f"{prefix}_{board}_{class_level}_{subject}_{stem}".replace(" ", "_"),
                textbook_upload_id=f"eval-{stem}",
            )
        )
    return EvalContext(
        config=cfg,
        run_id=rid,
        artifacts=artifacts,
        goldens=goldens,
        datasets=datasets,
    )

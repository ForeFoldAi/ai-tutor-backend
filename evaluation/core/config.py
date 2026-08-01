"""Evaluation config loader (YAML + env overrides)."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EVAL_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = EVAL_ROOT.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def default_config() -> dict[str, Any]:
    return {
        "datasets": {
            "root": str(EVAL_ROOT / "datasets"),
            "pdfs": ["chapter-1.pdf", "chapter-2.pdf"],
            "board": "CBSE",
            "class_level": "CLASS_9",
            "subject_name": "Social",
            "collection_prefix": "eval",
        },
        "paths": {
            "goldens": str(EVAL_ROOT / "goldens"),
            "artifacts": str(EVAL_ROOT / "artifacts"),
            "reports": str(EVAL_ROOT / "reports"),
            "baselines": str(EVAL_ROOT / "baselines"),
            "fixtures": str(EVAL_ROOT / "fixtures"),
            "expected": str(EVAL_ROOT / "expected"),
        },
        "phases": {
            "enabled": [
                "pdf_extraction",
                "images",
                "tables",
                "ocr",
                "chunking",
                "embeddings",
                "retrieval",
                "topics",
                "captions",
                "ai_tutor",
                "voice_tutor",
                "lesson_planner",
                "worksheet",
                "quiz",
                "homework",
                "science_experiments",
                "monitoring",
                "pipeline",
            ],
            "skip_llm": False,
            "skip_heavy_ml": False,
        },
        "thresholds": {
            "retrieval_top1": 0.6,
            "retrieval_top3": 0.8,
            "retrieval_mrr": 0.7,
            "retrieval_ndcg": 0.7,
            "hallucination_rate_max": 0.15,
            "groundedness_min": 0.7,
            "faithfulness_min": 0.7,
            "chunk_quality_min": 0.75,
            "image_quality_min": 0.8,
            "table_quality_min": 0.7,
            "caption_accuracy_min": 0.7,
            "lesson_quality_min": 0.7,
            "tutor_quality_min": 0.7,
            "embedding_dim": 768,
            "chunk_size_tokens_max": 1200,
            "chunk_size_tokens_min": 50,
            "latency_ms_warn": 30000,
            "latency_ms_fail": 120000,
            "blank_image_std_max": 8.0,
            "min_image_side_px": 40,
            "min_image_area_px": 2500,
            "crop_edge_margin_ratio": 0.02,
        },
        "regression": {
            "enabled": True,
            "max_score_drop": 0.05,
            "max_latency_increase_ratio": 1.5,
            "max_retrieval_drop": 0.05,
            "max_hallucination_increase": 0.05,
        },
        "ci": {
            "fail_on_critical": True,
            "fail_on_regression": True,
            "fail_on_overall_below": 0.85,
        },
        "llm_judge": {
            "enabled": True,
            "feature": "chat",
            "max_tokens": 512,
            "temperature": 0.0,
        },
        "runtime": {
            "parallel_datasets": False,
            "save_artifacts": True,
            "bootstrap_goldens": False,
        },
    }


class EvalConfig:
    def __init__(self, data: dict[str, Any] | None = None):
        self.data = _deep_merge(default_config(), data or {})

    @classmethod
    def load(cls, *extra_paths: str | Path, preset: str | None = None) -> EvalConfig:
        cfg = default_config()
        phases_yaml = load_yaml(EVAL_ROOT / "configs" / "phases.yaml")
        for name in ("default.yaml", "thresholds.yaml"):
            cfg = _deep_merge(cfg, load_yaml(EVAL_ROOT / "configs" / name))
        cfg = _deep_merge(cfg, {k: v for k, v in phases_yaml.items() if k != "presets"})
        for p in extra_paths:
            cfg = _deep_merge(cfg, load_yaml(Path(p)))
        # Resolve relative paths against backend root
        cfg = cls._resolve_paths(cfg)
        # Preset selection
        preset = preset or os.environ.get("EVAL_PRESET")
        if preset:
            presets = phases_yaml.get("presets") or {}
            if preset in presets:
                cfg["phases"]["enabled"] = list(presets[preset])
        # Env overrides
        if os.environ.get("EVAL_SKIP_LLM", "").lower() in ("1", "true", "yes"):
            cfg["phases"]["skip_llm"] = True
        if os.environ.get("EVAL_SKIP_HEAVY_ML", "").lower() in ("1", "true", "yes"):
            cfg["phases"]["skip_heavy_ml"] = True
        enabled = os.environ.get("EVAL_PHASES")
        if enabled:
            cfg["phases"]["enabled"] = [x.strip() for x in enabled.split(",") if x.strip()]
        return cls(cfg)

    @staticmethod
    def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
        def resolve(p: str) -> str:
            path = Path(p)
            if path.is_absolute():
                return str(path)
            # Prefer evaluation/… under EVAL_ROOT
            if p.startswith("evaluation/"):
                return str(BACKEND_ROOT / p)
            cand = EVAL_ROOT / p
            if cand.exists() or p.startswith("datasets") or p.startswith("goldens"):
                return str(EVAL_ROOT / p)
            return str(BACKEND_ROOT / p)

        if "datasets" in cfg and "root" in cfg["datasets"]:
            cfg["datasets"]["root"] = resolve(cfg["datasets"]["root"])
        for key in list((cfg.get("paths") or {}).keys()):
            cfg["paths"][key] = resolve(cfg["paths"][key])
        return cfg

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def threshold(self, name: str, default: float | None = None) -> float:
        v = self.get("thresholds", name, default=default)
        if v is None:
            raise KeyError(f"Missing threshold: {name}")
        return float(v)

    def pdf_paths(self) -> list[Path]:
        root = Path(self.get("datasets", "root"))
        return [root / name for name in self.get("datasets", "pdfs", default=[])]

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.data)

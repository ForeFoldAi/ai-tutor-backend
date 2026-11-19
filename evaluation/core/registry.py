"""Plugin registry for evaluation phases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from evaluation.core.context import EvalContext
    from evaluation.core.types import PhaseReport

_REGISTRY: dict[str, type[Phase]] = {}


class Phase(ABC):
    """Base class for a feature evaluation module."""

    phase_id: str = ""
    feature: str = ""
    critical: bool = False
    requires_llm: bool = False
    requires_heavy_ml: bool = False
    depends_on: list[str] = []

    @abstractmethod
    def run(self, ctx: EvalContext) -> PhaseReport:
        raise NotImplementedError


def register_phase(cls: type[Phase]) -> type[Phase]:
    if not cls.phase_id:
        raise ValueError(f"{cls.__name__} missing phase_id")
    _REGISTRY[cls.phase_id] = cls
    return cls


def get_phase(phase_id: str) -> type[Phase]:
    if phase_id not in _REGISTRY:
        raise KeyError(f"Unknown phase: {phase_id}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[phase_id]


def list_phases() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    _REGISTRY.clear()


def discover_phases() -> None:
    """Import all phase packages so @register_phase decorators fire."""
    # Local imports keep discovery lazy and avoid circular deps at package load.
    from evaluation.phases import (  # noqa: F401
        phase01_pdf_extraction,
        phase02_images,
        phase03_tables,
        phase04_ocr,
        phase05_chunking,
        phase06_embeddings,
        phase07_retrieval,
        phase08_topics,
        phase09_captions,
        phase10_ai_tutor,
        phase11_voice_tutor,
        phase12_lesson_planner,
        phase13_worksheet,
        phase14_quiz,
        phase15_homework,
        phase16_science_experiments,
        phase17_monitoring,
        phase18_pipeline,
        phase19_regression,
        phase20_reporting,
    )
    # Optional plugins
    import os
    import importlib

    for mod in (os.environ.get("EVAL_PLUGINS") or "").split(","):
        mod = mod.strip()
        if mod:
            importlib.import_module(mod)


def phase_factory(phase_id: str) -> Phase:
    return get_phase(phase_id)()


# Optional helper for one-off plugin scripts under evaluation/plugins/
def load_plugin(module_path: str) -> None:
    import importlib

    importlib.import_module(module_path)

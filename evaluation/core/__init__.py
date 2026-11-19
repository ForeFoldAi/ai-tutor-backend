"""Shared evaluation primitives."""

from evaluation.core.types import CheckResult, PhaseReport, Status
from evaluation.core.registry import Phase, get_phase, list_phases, register_phase

__all__ = [
    "CheckResult",
    "Phase",
    "PhaseReport",
    "Status",
    "get_phase",
    "list_phases",
    "register_phase",
]

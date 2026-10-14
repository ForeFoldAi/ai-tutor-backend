"""Shared pytest helpers that wrap phase runners."""

from __future__ import annotations

import pytest

from evaluation.core.config import EvalConfig
from evaluation.core.context import build_context
from evaluation.core.runner import run_phase
from evaluation.core.registry import discover_phases
from evaluation.core.types import Status


def run_feature(phase_id: str, *, preset: str | None = None, skip_llm=True, skip_heavy_ml=True):
    discover_phases()
    cfg = EvalConfig.load(preset=preset)
    cfg.data["phases"]["skip_llm"] = skip_llm
    cfg.data["phases"]["skip_heavy_ml"] = skip_heavy_ml
    # Force only this phase conceptually — still run it directly
    ctx = build_context(cfg)
    return run_phase(phase_id, ctx)


def assert_phase_ok(report, *, allow_skip=False):
    if report.status == Status.SKIP and allow_skip:
        return
    fails = [c for c in report.checks if c.status in (Status.FAIL, Status.ERROR) and c.critical]
    if fails:
        details = "\n".join(f"- {c.check_id}: {c.reason} (expected={c.expected} actual={c.actual})" for c in fails)
        pytest.fail(f"Phase {report.phase_id} critical failures:\n{details}")

"""Phase orchestration: dependency order, skips, aggregation."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Iterable

from evaluation.core.config import EvalConfig
from evaluation.core.context import EvalContext, build_context
from evaluation.core.registry import Phase, discover_phases, list_phases, phase_factory
from evaluation.core.types import PhaseReport, RunReport, Status, check


def _configure_eval_runtime(cfg: EvalConfig) -> None:
    """
    Keep eval isolated from production remote Qdrant timeouts unless opted in.

    ponytail: default local Chroma under evaluation/artifacts/_eval_chroma;
    set EVAL_USE_QDRANT=1 to hit the configured Qdrant instead.
    """
    use_qdrant = os.environ.get("EVAL_USE_QDRANT", "").lower() in ("1", "true", "yes")
    if not use_qdrant:
        os.environ["VECTOR_BACKEND"] = "chroma"
        try:
            import app.config as app_cfg

            app_cfg.VECTOR_BACKEND = "chroma"
            chroma_path = Path(cfg.get("paths", "artifacts")) / "_eval_chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            app_cfg.CHROMA_PATH = str(chroma_path)
            os.environ["CHROMA_PATH"] = str(chroma_path)
        except Exception:
            pass


def _topo_sort(phase_ids: list[str]) -> list[str]:
    discover_phases()
    remaining = set(phase_ids)
    ordered: list[str] = []
    while remaining:
        progressed = False
        for pid in sorted(remaining):
            cls = phase_factory(pid).__class__
            deps = [d for d in (cls.depends_on or []) if d in remaining]
            if not deps:
                ordered.append(pid)
                remaining.remove(pid)
                progressed = True
        if not progressed:
            # Cycle or missing dep — append rest alphabetically
            ordered.extend(sorted(remaining))
            break
    return ordered


def _should_skip(phase: Phase, ctx: EvalContext) -> str | None:
    if phase.requires_llm and ctx.skip_llm():
        return "LLM skipped via config/env"
    if phase.requires_heavy_ml and ctx.skip_heavy_ml():
        return "Heavy ML skipped via config/env"
    return None


def run_phase(phase_id: str, ctx: EvalContext) -> PhaseReport:
    phase = phase_factory(phase_id)
    skip_reason = _should_skip(phase, ctx)
    if skip_reason:
        report = PhaseReport(
            phase_id=phase.phase_id,
            feature=phase.feature,
            status=Status.SKIP,
            checks=[
                check(
                    phase.feature,
                    "skipped",
                    ok=True,
                    skip=True,
                    reason=skip_reason,
                )
            ],
        )
        return report.finalize()

    t0 = time.perf_counter()
    try:
        report = phase.run(ctx)
    except Exception as exc:  # noqa: BLE001 — surface as ERROR check
        report = PhaseReport(
            phase_id=phase.phase_id,
            feature=phase.feature or phase_id,
            status=Status.ERROR,
            failed_stage=phase_id,
            checks=[
                check(
                    phase.feature or phase_id,
                    "unhandled_exception",
                    ok=False,
                    error=True,
                    critical=phase.critical,
                    reason=f"{type(exc).__name__}: {exc}",
                    evidence={"traceback": traceback.format_exc()},
                )
            ],
        )
    report.execution_ms = (time.perf_counter() - t0) * 1000
    return report.finalize()


def run_evaluation(
    *,
    config: EvalConfig | None = None,
    phases: Iterable[str] | None = None,
    run_id: str | None = None,
) -> RunReport:
    discover_phases()
    cfg = config or EvalConfig.load()
    _configure_eval_runtime(cfg)
    ctx = build_context(cfg, run_id=run_id)
    selected = list(phases) if phases is not None else list(cfg.get("phases", "enabled", default=[]))
    # Drop unknown ids early with clear ERROR phase
    known = set(list_phases())
    ordered = _topo_sort([p for p in selected if p in known])
    report = RunReport(
        run_id=ctx.run_id,
        started_at=ctx.started_at,
        config_snapshot=cfg.snapshot(),
    )
    for pid in selected:
        if pid not in known:
            report.phases.append(
                PhaseReport(
                    phase_id=pid,
                    feature=pid,
                    status=Status.ERROR,
                    checks=[
                        check(
                            pid,
                            "unknown_phase",
                            ok=False,
                            error=True,
                            critical=True,
                            reason=f"Phase '{pid}' is not registered",
                            expected=sorted(known),
                        )
                    ],
                ).finalize()
            )
    for pid in ordered:
        report.phases.append(run_phase(pid, ctx))
        ctx.cache[f"phase_report:{pid}"] = report.phases[-1]

    # Optional regression compare
    if cfg.get("regression", "enabled", default=True):
        from evaluation.phases.phase19_regression.compare import attach_regression

        attach_regression(report, cfg)

    report.finalize()
    ctx.artifacts.write_json("run_report.json", report.to_dict())
    return report

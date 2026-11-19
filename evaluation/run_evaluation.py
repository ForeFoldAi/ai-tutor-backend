#!/usr/bin/env python3
"""CLI entrypoint: run the AI evaluation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.core.config import EvalConfig
from evaluation.core.runner import run_evaluation
from evaluation.core.types import Status
from evaluation.metrics.aggregate import overall_ai_quality_score
from evaluation.phases.phase20_reporting.render import render_all, to_console


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Tutor Evaluation Framework")
    parser.add_argument("--phases", help="Comma-separated phase ids (overrides config)")
    parser.add_argument("--preset", help="Preset from configs/phases.yaml (smoke|extraction|rag|teacher|full)")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-heavy-ml", action="store_true")
    parser.add_argument("--bootstrap-goldens", action="store_true")
    parser.add_argument("--config", action="append", default=[], help="Extra YAML config path")
    parser.add_argument("--list-phases", action="store_true")
    args = parser.parse_args(argv)

    if args.list_phases:
        from evaluation.core.registry import discover_phases, list_phases

        discover_phases()
        for p in list_phases():
            print(p)
        return 0

    cfg = EvalConfig.load(*args.config, preset=args.preset)
    if args.skip_llm:
        cfg.data["phases"]["skip_llm"] = True
    if args.skip_heavy_ml:
        cfg.data["phases"]["skip_heavy_ml"] = True
    if args.bootstrap_goldens:
        cfg.data["runtime"]["bootstrap_goldens"] = True

    phases = None
    if args.phases:
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    report = run_evaluation(config=cfg, phases=phases)
    scores = overall_ai_quality_score(report)
    paths = render_all(report, cfg, extra=scores)
    print(to_console(report.to_dict() | {"overall_ai_quality": scores}))
    print("Reports:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    # CI exit codes
    ci = cfg.get("ci", default={}) or {}
    if ci.get("fail_on_critical") and any(p.critical_failures for p in report.phases):
        return 2
    if ci.get("fail_on_regression") and (report.regression or {}).get("regressed"):
        return 3
    min_score = float(ci.get("fail_on_overall_below", 0.0) or 0.0)
    # Don't gate on overall score when most phases were intentionally skipped
    decisive = sum(
        1
        for p in report.phases
        for c in p.checks
        if c.status.value in ("PASS", "FAIL", "ERROR", "WARN")
    )
    if min_score and decisive >= 5 and report.overall_score < min_score:
        return 4
    if report.overall_status in (Status.FAIL, Status.ERROR):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

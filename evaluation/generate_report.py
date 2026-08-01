#!/usr/bin/env python3
"""Render reports from an existing run_report.json or latest artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.core.config import EvalConfig
from evaluation.core.types import PhaseReport, RunReport, Status, CheckResult
from evaluation.metrics.aggregate import overall_ai_quality_score
from evaluation.phases.phase20_reporting.render import render_all, to_console


def _load_run(path: Path) -> RunReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    phases = []
    for p in data.get("phases", []):
        checks = [
            CheckResult(
                feature=c["feature"],
                check_id=c["check_id"],
                status=Status(c["status"]),
                reason=c.get("reason", ""),
                expected=c.get("expected"),
                actual=c.get("actual"),
                evidence=c.get("evidence") or {},
                execution_ms=float(c.get("execution_ms") or 0),
                confidence=float(c.get("confidence") or 1),
                critical=bool(c.get("critical")),
                metric_name=c.get("metric_name"),
                metric_value=c.get("metric_value"),
            )
            for c in p.get("checks", [])
        ]
        phases.append(
            PhaseReport(
                phase_id=p["phase_id"],
                feature=p["feature"],
                status=Status(p["status"]),
                checks=checks,
                metrics=p.get("metrics") or {},
                artifacts=p.get("artifacts") or [],
                execution_ms=float(p.get("execution_ms") or 0),
                failed_stage=p.get("failed_stage"),
                dataset=p.get("dataset"),
                meta=p.get("meta") or {},
            )
        )
    run = RunReport(
        run_id=data["run_id"],
        started_at=float(data.get("started_at") or 0),
        finished_at=float(data.get("finished_at") or 0),
        phases=phases,
        overall_status=Status(data.get("overall_status", "PASS")),
        overall_score=float(data.get("overall_score") or 0),
        regression=data.get("regression") or {},
        config_snapshot=data.get("config_snapshot") or {},
    )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("--from", dest="source", help="Path to run_report.json")
    parser.add_argument("--latest", action="store_true", help="Use reports/latest.json")
    args = parser.parse_args(argv)
    cfg = EvalConfig.load()
    if args.latest:
        src = Path(cfg.get("paths", "reports")) / "latest.json"
    elif args.source:
        src = Path(args.source)
    else:
        # Prefer newest artifacts/*/run_report.json
        arts = Path(cfg.get("paths", "artifacts"))
        candidates = sorted(arts.glob("*/run_report.json"))
        if not candidates:
            print("No run_report.json found. Pass --from or --latest.", file=sys.stderr)
            return 1
        src = candidates[-1]

    run = _load_run(src)
    scores = overall_ai_quality_score(run)
    paths = render_all(run, cfg, extra=scores)
    print(to_console(run.to_dict() | {"overall_ai_quality": scores}))
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

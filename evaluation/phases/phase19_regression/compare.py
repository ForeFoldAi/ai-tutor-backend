"""Regression compare helpers (imported by runner)."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.core.types import RunReport, Status


def _baseline_dir(cfg) -> Path:
    return Path(cfg.get("paths", "baselines"))


def latest_baseline(cfg) -> dict | None:
    root = _baseline_dir(cfg)
    latest = root / "latest.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    if not root.exists():
        return None
    files = sorted(root.glob("baseline_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def save_baseline(run: RunReport, cfg) -> Path:
    root = _baseline_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"baseline_{run.run_id}.json"
    payload = json.dumps(run.to_dict(), indent=2, default=str)
    path.write_text(payload, encoding="utf-8")
    (root / "latest.json").write_text(payload, encoding="utf-8")
    return path


def compare_runs(current: RunReport, previous: dict, cfg) -> dict:
    reg = cfg.get("regression", default={}) or {}
    max_drop = float(reg.get("max_score_drop", 0.05))
    max_lat = float(reg.get("max_latency_increase_ratio", 1.5))
    max_ret = float(reg.get("max_retrieval_drop", 0.05))
    max_hall = float(reg.get("max_hallucination_increase", 0.05))

    cur_ids = {p.phase_id for p in current.phases}
    prev_ids = {p["phase_id"] for p in previous.get("phases", [])}
    # Don't treat smoke→full as regression noise
    if cur_ids != prev_ids and len(cur_ids.symmetric_difference(prev_ids)) > 2:
        return {
            "baseline_run_id": previous.get("run_id"),
            "regressions": [],
            "regressed": False,
            "note": "Baseline phase set differs — skipped regression compare",
        }

    prev_phases = {p["phase_id"]: p for p in previous.get("phases", [])}
    regressions = []
    for p in current.phases:
        prev = prev_phases.get(p.phase_id)
        if not prev:
            continue
        if p.status.value == "SKIP" or (prev.get("status") == "SKIP"):
            continue
        cur_score = float(p.metrics.get("quality_score", p.passed / max(1, len(p.checks))))
        prev_score = float((prev.get("metrics") or {}).get("quality_score", 0.0))
        if prev_score and cur_score < prev_score - max_drop:
            regressions.append(
                {
                    "type": "quality_drop",
                    "phase": p.phase_id,
                    "previous": prev_score,
                    "current": cur_score,
                    "drop": prev_score - cur_score,
                }
            )
        if p.phase_id == "retrieval":
            for key in ("retrieval_top1", "retrieval_top3", "retrieval_mrr", "retrieval_ndcg"):
                if key in p.metrics and key in (prev.get("metrics") or {}):
                    drop = float(prev["metrics"][key]) - float(p.metrics[key])
                    if drop > max_ret:
                        regressions.append(
                            {
                                "type": "retrieval_degradation",
                                "metric": key,
                                "previous": prev["metrics"][key],
                                "current": p.metrics[key],
                                "drop": drop,
                            }
                        )
        if p.phase_id == "ai_tutor":
            prev_h = float((prev.get("metrics") or {}).get("hallucination_rate", 0))
            cur_h = float(p.metrics.get("hallucination_rate", 0))
            if cur_h - prev_h > max_hall:
                regressions.append(
                    {
                        "type": "hallucination_increase",
                        "previous": prev_h,
                        "current": cur_h,
                    }
                )
        prev_ms = float(prev.get("execution_ms") or 0)
        # Ignore micro-phase jitter; only flag meaningful latency regressions
        if prev_ms >= 5000 and p.execution_ms > prev_ms * max_lat:
            regressions.append(
                {
                    "type": "latency_increase",
                    "phase": p.phase_id,
                    "previous_ms": prev_ms,
                    "current_ms": p.execution_ms,
                    "ratio": p.execution_ms / prev_ms,
                }
            )
    return {
        "baseline_run_id": previous.get("run_id"),
        "regressions": regressions,
        "regressed": len(regressions) > 0,
    }


def attach_regression(run: RunReport, cfg) -> None:
    prev = latest_baseline(cfg)
    if not prev:
        run.regression = {"regressed": False, "note": "No baseline yet — current run becomes baseline"}
        save_baseline(run, cfg)
        return
    cmp = compare_runs(run, prev, cfg)
    run.regression = cmp
    if run.overall_status == Status.PASS and not cmp.get("regressed"):
        save_baseline(run, cfg)

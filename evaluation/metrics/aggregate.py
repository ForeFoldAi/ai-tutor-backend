"""Aggregate overall AI quality score from phase metrics."""

from __future__ import annotations

from typing import Any

from evaluation.core.types import PhaseReport, RunReport


WEIGHTS = {
    "pdf_extraction": 0.08,
    "images": 0.10,
    "tables": 0.06,
    "ocr": 0.04,
    "chunking": 0.08,
    "embeddings": 0.06,
    "retrieval": 0.14,
    "topics": 0.04,
    "captions": 0.05,
    "ai_tutor": 0.14,
    "lesson_planner": 0.07,
    "quiz": 0.04,
    "worksheet": 0.03,
    "homework": 0.03,
    "monitoring": 0.04,
}


def phase_score(report: PhaseReport) -> float:
    if report.metrics.get("quality_score") is not None:
        return float(report.metrics["quality_score"])
    if not report.checks:
        return 0.0
    return report.passed / len(report.checks)


def overall_ai_quality_score(run: RunReport) -> dict[str, Any]:
    by_id = {p.phase_id: p for p in run.phases}
    weighted = 0.0
    total_w = 0.0
    breakdown = {}
    for pid, w in WEIGHTS.items():
        if pid not in by_id:
            continue
        s = phase_score(by_id[pid])
        breakdown[pid] = s
        weighted += w * s
        total_w += w
    overall = (weighted / total_w) if total_w else run.overall_score
    return {"overall_ai_quality_score": overall, "breakdown": breakdown}

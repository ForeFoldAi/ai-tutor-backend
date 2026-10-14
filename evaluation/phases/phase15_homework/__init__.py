"""Phase 15 — Homework generator evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._artifact_eval import evaluate_artifact_kind
from evaluation.phases._helpers import new_report


def _homework_quality(payload: dict, case: dict) -> dict:
    sections = [
        "practice_questions",
        "observation_tasks",
        "project_work",
        "reading_assignment",
        "revision_tasks",
    ]
    counts = {s: len(payload.get(s) or []) for s in sections}
    total = sum(counts.values())
    checks = [
        {
            "id": "has_tasks",
            "ok": total >= int(case.get("min_tasks") or 2),
            "critical": True,
            "reason": "Homework lacks tasks",
            "actual": counts,
        },
        {
            "id": "practice_present",
            "ok": counts["practice_questions"] >= 1 or total >= 3,
            "reason": "No practice questions",
            "actual": counts["practice_questions"],
        },
    ]
    score = sum(1 for c in checks if c["ok"]) / len(checks)
    return {"score": score, "checks": checks}


@register_phase
class HomeworkPhase(Phase):
    phase_id = "homework"
    feature = "homework"
    requires_llm = True

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        evaluate_artifact_kind(ctx, report, kind="homework", feature=self.feature, quality_checks=_homework_quality)
        return report.finalize()

"""Phase 13 — Worksheet generator evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._artifact_eval import evaluate_artifact_kind
from evaluation.phases._helpers import new_report


def _worksheet_quality(payload: dict, case: dict) -> dict:
    sections = [
        "fill_blanks",
        "true_false",
        "match_following",
        "short_answer",
        "long_answer",
        "application_questions",
    ]
    counts = {s: len(payload.get(s) or []) for s in sections}
    total = sum(counts.values())
    checks = [
        {
            "id": "has_questions",
            "ok": total >= int(case.get("min_questions") or 3),
            "critical": True,
            "reason": "Too few worksheet questions",
            "expected": case.get("min_questions") or 3,
            "actual": counts,
        },
        {
            "id": "coverage_variety",
            "ok": sum(1 for v in counts.values() if v > 0) >= 2,
            "reason": "Worksheet lacks question-type variety",
            "actual": counts,
        },
    ]
    score = sum(1 for c in checks if c["ok"]) / len(checks)
    return {"score": score, "checks": checks}


@register_phase
class WorksheetPhase(Phase):
    phase_id = "worksheet"
    feature = "worksheet"
    requires_llm = True

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        evaluate_artifact_kind(ctx, report, kind="worksheet", feature=self.feature, quality_checks=_worksheet_quality)
        return report.finalize()

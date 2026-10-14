"""Phase 14 — Quiz generator evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._artifact_eval import evaluate_artifact_kind
from evaluation.phases._helpers import new_report


def _quiz_quality(payload: dict, case: dict) -> dict:
    mcq = payload.get("mcq") or []
    checks = [
        {
            "id": "has_mcq",
            "ok": len(mcq) >= int(case.get("min_mcq") or 1),
            "critical": True,
            "reason": "Quiz missing MCQs",
            "actual": len(mcq),
        }
    ]
    # Duplicate questions
    texts = [str(q.get("question") or "").strip().lower() for q in mcq]
    dup = len(texts) - len(set(texts))
    checks.append(
        {
            "id": "no_duplicates",
            "ok": dup == 0,
            "critical": True,
            "reason": "Duplicate quiz questions",
            "actual": dup,
        }
    )
    # Distractors
    for i, q in enumerate(mcq):
        opts = q.get("options") or []
        checks.append(
            {
                "id": f"mcq{i}.distractors",
                "ok": len(opts) >= 3,
                "reason": "MCQ needs >=3 options",
                "actual": len(opts),
            }
        )
        ans = q.get("answer")
        if ans is not None and opts:
            checks.append(
                {
                    "id": f"mcq{i}.answer_in_options",
                    "ok": str(ans) in [str(o) for o in opts] or ans in (0, 1, 2, 3, "A", "B", "C", "D", "a", "b", "c", "d"),
                    "critical": True,
                    "reason": "Correct answer not aligned with options",
                    "expected": ans,
                    "actual": opts,
                }
            )
    if payload.get("answers") is not None:
        checks.append(
            {
                "id": "answers_key",
                "ok": isinstance(payload.get("answers"), dict),
                "reason": "answers must be a dict",
            }
        )
    score = sum(1 for c in checks if c["ok"]) / max(1, len(checks))
    return {"score": score, "checks": checks}


@register_phase
class QuizPhase(Phase):
    phase_id = "quiz"
    feature = "quiz"
    critical = True
    requires_llm = True

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        evaluate_artifact_kind(ctx, report, kind="quiz", feature=self.feature, quality_checks=_quiz_quality)
        return report.finalize()

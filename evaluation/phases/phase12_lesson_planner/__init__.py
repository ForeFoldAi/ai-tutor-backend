"""Phase 12 — Lesson planner evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._artifact_eval import evaluate_artifact_kind
from evaluation.phases._helpers import new_report


def _lesson_quality(payload: dict, case: dict) -> dict:
    checks = []
    objs = payload.get("objectives") or []
    acts = payload.get("activities") or []
    duration = int(payload.get("duration") or 0)
    checks.append(
        {
            "id": "objectives",
            "ok": len(objs) >= 1,
            "critical": True,
            "reason": "Learning objectives required",
            "actual": len(objs),
        }
    )
    checks.append(
        {
            "id": "activities",
            "ok": len(acts) >= 1,
            "critical": True,
            "reason": "Activities required",
            "actual": len(acts),
        }
    )
    checks.append(
        {
            "id": "assessment",
            "ok": bool(payload.get("assessment_plan")),
            "reason": "Assessment plan missing",
            "actual": payload.get("assessment_plan"),
        }
    )
    checks.append(
        {
            "id": "time_allocation",
            "ok": duration > 0 and sum(int(a.get("duration_minutes") or 0) for a in acts) <= duration + 10,
            "reason": "Activity time exceeds lesson duration",
            "expected": duration,
            "actual": sum(int(a.get("duration_minutes") or 0) for a in acts),
        }
    )
    bloom = case.get("bloom_keywords") or ["understand", "explain", "identify", "apply", "analyse", "analyze", "create"]
    blob = " ".join(objs).lower()
    bloom_hit = any(b in blob for b in bloom)
    checks.append(
        {
            "id": "blooms_signal",
            "ok": bloom_hit or len(objs) >= 2,
            "reason": "Objectives lack Bloom-style verbs",
            "expected": bloom,
            "actual": objs,
        }
    )
    topic = (case.get("topic") or "").lower()
    subject_ok = topic in (payload.get("title") or "").lower() or any(topic in o.lower() for o in objs) or not topic
    checks.append(
        {
            "id": "subject_topic",
            "ok": subject_ok,
            "critical": True,
            "reason": "Lesson not aligned to topic",
            "expected": case.get("topic"),
            "actual": payload.get("title"),
        }
    )
    score = sum(1 for c in checks if c["ok"]) / max(1, len(checks))
    return {"score": score, "checks": checks}


@register_phase
class LessonPlannerPhase(Phase):
    phase_id = "lesson_planner"
    feature = "lesson_planner"
    critical = True
    requires_llm = True

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        # Also validate teaching_notes schema path
        evaluate_artifact_kind(ctx, report, kind="lesson_plan", feature=self.feature, quality_checks=_lesson_quality)
        return report.finalize()

"""Phase 18 — End-to-end pipeline validation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport, Status
from evaluation.phases._helpers import add, new_report

# Ordered stages — failure short-circuits remaining stages in the report narrative.
PIPELINE_STAGES = [
    "pdf_extraction",
    "ocr",
    "images",
    "tables",
    "chunking",
    "embeddings",
    "retrieval",
    "ai_tutor",
    "lesson_planner",
    "quiz",
    "worksheet",
    "homework",
    "monitoring",
]


@register_phase
class PipelinePhase(Phase):
    phase_id = "pipeline"
    feature = "pipeline"
    critical = True
    depends_on = list(PIPELINE_STAGES)

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        failed_stage = None
        stage_status = {}

        for stage in PIPELINE_STAGES:
            prev = ctx.cache.get(f"phase_report:{stage}")
            if prev is None:
                # Stage not executed in this run — skip rather than fail hard
                stage_status[stage] = "SKIP"
                add(
                    report,
                    feature=self.feature,
                    check_id=f"stage.{stage}",
                    ok=True,
                    skip=True,
                    reason="Stage not included in this run",
                )
                continue
            status = prev.status
            stage_status[stage] = status.value
            ok = (
                status in (Status.PASS, Status.WARN, Status.SKIP)
                or (status == Status.FAIL and not prev.critical_failures)
            )
            if not ok and failed_stage is None:
                failed_stage = stage
            add(
                report,
                feature=self.feature,
                check_id=f"stage.{stage}",
                ok=ok,
                critical=True,
                reason=(
                    f"Pipeline stage '{stage}' {status.value}"
                    + (f" — first failure" if failed_stage == stage else "")
                ),
                expected="PASS",
                actual=status.value,
                evidence={
                    "passed": prev.passed,
                    "failed": prev.failed,
                    "metrics": prev.metrics,
                    "critical_failures": [c.check_id for c in prev.critical_failures],
                },
            )

        report.failed_stage = failed_stage
        report.meta["stage_status"] = stage_status
        report.metrics["quality_score"] = (
            0.0
            if failed_stage
            else (report.passed / len(report.checks) if report.checks else 0.0)
        )
        ctx.artifacts.write_json("pipeline/stages.json", {"failed_stage": failed_stage, "stages": stage_status})
        if failed_stage:
            add(
                report,
                feature=self.feature,
                check_id="pipeline.overall",
                ok=False,
                critical=True,
                reason=f"End-to-end pipeline FAILED at stage: {failed_stage}",
                expected="all stages PASS",
                actual=failed_stage,
            )
        else:
            add(
                report,
                feature=self.feature,
                check_id="pipeline.overall",
                ok=True,
                reason="All executed pipeline stages passed",
                actual=stage_status,
            )
        return report.finalize()

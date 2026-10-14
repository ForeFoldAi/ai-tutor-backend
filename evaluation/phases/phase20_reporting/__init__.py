"""Phase 20 — Reporting (also invoked via generate_report.py)."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, new_report
from evaluation.phases.phase20_reporting.render import render_all


@register_phase
class ReportingPhase(Phase):
    phase_id = "reporting"
    feature = "reporting"
    depends_on = ["regression"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        # Collect prior phase reports into a synthetic run for rendering mid-suite.
        from evaluation.core.types import RunReport
        from evaluation.metrics.aggregate import overall_ai_quality_score

        run = RunReport(run_id=ctx.run_id, started_at=ctx.started_at, phases=[])
        for key, val in ctx.cache.items():
            if key.startswith("phase_report:"):
                run.phases.append(val)
        run.finalize()
        scores = overall_ai_quality_score(run)
        run.config_snapshot = ctx.config.snapshot()
        paths = render_all(run, ctx.config, extra=scores)
        for label, path in paths.items():
            report.artifacts.append(path)
            add(
                report,
                feature=self.feature,
                check_id=f"wrote.{label}",
                ok=True,
                reason=f"Wrote {label} report",
                actual=path,
            )
        report.metrics["quality_score"] = float(scores.get("overall_ai_quality_score") or 0)
        report.meta["overall_ai_quality_score"] = scores
        return report.finalize()

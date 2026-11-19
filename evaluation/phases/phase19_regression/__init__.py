"""Phase 19 — Regression comparison vs stored baselines."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport, RunReport
from evaluation.phases._helpers import add, new_report
from evaluation.phases.phase19_regression.compare import compare_runs, latest_baseline


@register_phase
class RegressionPhase(Phase):
    phase_id = "regression"
    feature = "regression"
    depends_on = ["pipeline"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        prev = latest_baseline(ctx.config)
        if not prev:
            add(
                report,
                feature=self.feature,
                check_id="baseline_exists",
                ok=True,
                skip=True,
                reason="No previous baseline; will be created after run",
            )
            return report.finalize()

        add(
            report,
            feature=self.feature,
            check_id="baseline_loaded",
            ok=True,
            reason=f"Loaded baseline {prev.get('run_id')}",
            actual=prev.get("run_id"),
        )
        fake = RunReport(run_id=ctx.run_id, started_at=ctx.started_at, phases=[])
        for key, val in list(ctx.cache.items()):
            if key.startswith("phase_report:"):
                fake.phases.append(val)
        cmp = compare_runs(fake, prev, ctx.config)
        ctx.artifacts.write_json("regression/compare.json", cmp)
        add(
            report,
            feature=self.feature,
            check_id="no_regressions",
            ok=not cmp.get("regressed"),
            critical=True,
            reason="Quality/performance regressions detected" if cmp.get("regressed") else "No regressions",
            actual=cmp.get("regressions"),
        )
        report.metrics["quality_score"] = 0.0 if cmp.get("regressed") else 1.0
        report.meta["regression"] = cmp
        return report.finalize()

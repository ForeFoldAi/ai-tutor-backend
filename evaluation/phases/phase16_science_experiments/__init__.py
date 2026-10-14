"""Phase 16 — Science experiments evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, new_report


@register_phase
class ScienceExperimentsPhase(Phase):
    phase_id = "science_experiments"
    feature = "science_experiments"

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        try:
            from app.services.science_experiment import schemas as sci_schemas
            from app.services.science_experiment.experiment_catalog import (  # type: ignore
                ExperimentCatalog,
            )
        except Exception:
            try:
                from app.services.science_experiment import schemas as sci_schemas

                ExperimentCatalog = None  # type: ignore
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id="import",
                    ok=False,
                    error=True,
                    critical=True,
                    reason=str(exc),
                )
                return report.finalize()

        # Schema must exist
        model = getattr(sci_schemas, "ScienceExperiment", None)
        add(
            report,
            feature=self.feature,
            check_id="schema_present",
            ok=model is not None,
            critical=True,
            reason="ScienceExperiment schema missing",
        )

        for ds in ctx.datasets:
            golden = ctx.goldens.load(ds.name, "science_experiments", default={}) or {}
            cases = golden.get("cases") or []
            if not cases:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.cases",
                    ok=True,
                    skip=True,
                    reason="No science experiment goldens; structural checks only",
                )
                continue
            for i, case in enumerate(cases):
                query = case.get("query") or case.get("topic")
                # Prefer catalog / service match
                matched = None
                try:
                    from app.services.science_experiment.service import (  # type: ignore
                        match_experiment,
                    )

                    matched = match_experiment(query)
                except Exception:
                    matched = None
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.match",
                    ok=matched is not None or not case.get("must_match", True),
                    reason="No experiment matched for query",
                    expected=case.get("expected_id"),
                    actual=str(matched)[:300] if matched else None,
                )
                if matched is not None and model is not None:
                    try:
                        if hasattr(matched, "model_dump"):
                            data = matched.model_dump()
                        elif isinstance(matched, dict):
                            data = matched
                        else:
                            data = dict(matched)
                        model.model_validate(data)
                        ok = True
                        reason = "schema_ok"
                    except Exception as exc:  # noqa: BLE001
                        ok = False
                        reason = str(exc)
                        data = None
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.case{i}.schema",
                        ok=ok,
                        critical=True,
                        reason=reason,
                    )
                    if data:
                        ctx.artifacts.write_json(f"{ds.name}/science/case_{i}.json", data)

        report.metrics["quality_score"] = (
            report.passed / len(report.checks) if report.checks else 0.0
        )
        return report.finalize()

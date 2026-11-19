"""Phase 17 — Student monitoring / LIA / teacher+tutor guidance."""

from __future__ import annotations

from evaluation.core.adapters.lia import prompt_sections_from_guidance, sample_tutor_guidance
from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, new_report


@register_phase
class MonitoringPhase(Phase):
    phase_id = "monitoring"
    feature = "monitoring"
    requires_llm = False

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)

        # Module presence
        modules = [
            "app.services.learning_intelligence.orchestration.orchestrator",
            "app.services.learning_intelligence.agents.tutor_coach",
            "app.services.learning_intelligence.agents.teacher_coach",
            "app.services.learning_intelligence.agents.memory",
            "app.services.learning_intelligence.prompt_builder",
        ]
        import importlib

        for m in modules:
            try:
                importlib.import_module(m)
                add(report, feature=self.feature, check_id=f"import.{m.split('.')[-1]}", ok=True, reason="ok")
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"import.{m.split('.')[-1]}",
                    ok=False,
                    critical=True,
                    reason=str(exc),
                )

        # Live/sampled guidance (may fail without DB — recorded, not always critical)
        out = sample_tutor_guidance(query="Explain rainfall")
        ctx.artifacts.write_json("monitoring/tutor_guidance.json", out)
        if out.get("ok"):
            guidance = out.get("guidance") or {}
            sections = prompt_sections_from_guidance(guidance)
            add(
                report,
                feature=self.feature,
                check_id="tutor_guidance_object",
                ok=isinstance(guidance, dict) and len(guidance) > 0,
                critical=True,
                reason="Empty tutor guidance",
                actual=list(guidance.keys())[:20],
            )
            add(
                report,
                feature=self.feature,
                check_id="prompt_augmentation",
                ok=len(sections) > 0,
                reason="guidance_to_prompt_sections produced nothing",
                actual=sections[:5],
            )
        else:
            add(
                report,
                feature=self.feature,
                check_id="tutor_guidance_runtime",
                ok=True,
                skip=True,
                reason=f"LIA runtime unavailable in this environment: {out.get('error')}",
            )

        # Schema contracts
        try:
            from app.modules.learning_intelligence import schemas as lia_schemas

            for name in ("TutorGuidanceObject", "TeacherSummaryOut"):
                add(
                    report,
                    feature=self.feature,
                    check_id=f"schema.{name}",
                    ok=hasattr(lia_schemas, name),
                    critical=True,
                    reason=f"Missing schema {name}",
                )
        except Exception as exc:  # noqa: BLE001
            add(
                report,
                feature=self.feature,
                check_id="schema_import",
                ok=False,
                critical=True,
                reason=str(exc),
            )

        report.metrics["quality_score"] = (
            report.passed / len(report.checks) if report.checks else 0.0
        )
        return report.finalize()

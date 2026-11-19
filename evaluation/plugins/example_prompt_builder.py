"""
Example plugin — register a custom phase without editing core.

Usage:
  PYTHONPATH=. python -c "import evaluation.plugins.example_prompt_builder; ..."
  or set EVAL_PLUGINS=evaluation.plugins.example_prompt_builder
"""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, new_report


@register_phase
class PromptBuilderPhase(Phase):
    phase_id = "prompt_builder"
    feature = "prompt_builder"

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        try:
            from app.services.learning_intelligence import prompt_builder as pb

            add(
                report,
                feature=self.feature,
                check_id="module_import",
                ok=True,
                reason="prompt_builder importable",
            )
            add(
                report,
                feature=self.feature,
                check_id="has_guidance_to_prompt_sections",
                ok=hasattr(pb, "guidance_to_prompt_sections"),
                critical=True,
                reason="guidance_to_prompt_sections missing",
            )
        except Exception as exc:  # noqa: BLE001
            add(
                report,
                feature=self.feature,
                check_id="module_import",
                ok=False,
                error=True,
                critical=True,
                reason=str(exc),
            )
        return report.finalize()

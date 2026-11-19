"""Phase 11 — Voice tutor structural / prompt evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, new_report


@register_phase
class VoiceTutorPhase(Phase):
    phase_id = "voice_tutor"
    feature = "voice_tutor"
    requires_llm = False

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        # Structural checks against production voice modules (no live mic in CI)
        modules = [
            ("app.services.voice_tutor", "voice_tutor"),
            ("app.services.voice_prompts", "voice_prompts"),
            ("app.services.voice_chunking", "voice_chunking"),
            ("app.services.voice_tts_orchestrator", "voice_tts_orchestrator"),
            ("app.voice_ws", "voice_ws"),
        ]
        import importlib

        for mod_name, label in modules:
            try:
                importlib.import_module(mod_name)
                add(
                    report,
                    feature=self.feature,
                    check_id=f"import.{label}",
                    ok=True,
                    reason=f"{mod_name} importable",
                )
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"import.{label}",
                    ok=False,
                    critical=True,
                    reason=str(exc),
                )

        # Prompt contract: voice prompts should discourage long markdown / unsafe content
        try:
            from app.services import voice_prompts as vp

            text = ""
            for attr in dir(vp):
                val = getattr(vp, attr, None)
                if isinstance(val, str) and len(val) > 40:
                    text += "\n" + val
            add(
                report,
                feature=self.feature,
                check_id="prompt_presence",
                ok=len(text) > 100,
                reason="Voice prompts appear empty",
                actual=len(text),
            )
            # Chunking helper exists
            from app.services import voice_chunking as vc

            if hasattr(vc, "chunk_for_tts") or hasattr(vc, "split_for_tts"):
                fn = getattr(vc, "chunk_for_tts", None) or getattr(vc, "split_for_tts")
                parts = fn("Hello student. This is a short voice answer about rainfall.")
                add(
                    report,
                    feature=self.feature,
                    check_id="tts_chunking",
                    ok=isinstance(parts, (list, tuple)) and len(list(parts)) >= 1,
                    reason="TTS chunking returned empty",
                    actual=str(parts)[:200],
                )
            else:
                add(
                    report,
                    feature=self.feature,
                    check_id="tts_chunking",
                    ok=True,
                    skip=True,
                    reason="No chunk_for_tts/split_for_tts helper found",
                )
        except Exception as exc:  # noqa: BLE001
            add(
                report,
                feature=self.feature,
                check_id="voice_prompt_contract",
                ok=False,
                reason=str(exc),
            )

        report.metrics["quality_score"] = (
            report.passed / len(report.checks) if report.checks else 0.0
        )
        return report.finalize()

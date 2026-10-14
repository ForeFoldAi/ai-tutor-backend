import pytest

from evaluation.core.config import EvalConfig
from evaluation.core.runner import run_evaluation
from evaluation.tests.conftest import assert_phase_ok


@pytest.mark.eval
@pytest.mark.pipeline
def test_pipeline_smoke():
    cfg = EvalConfig.load(preset="smoke")
    cfg.data["phases"]["skip_llm"] = True
    cfg.data["phases"]["skip_heavy_ml"] = True
    # Ensure pipeline + reporting included
    enabled = list(cfg.get("phases", "enabled") or [])
    for extra in ("pipeline", "reporting", "voice_tutor", "monitoring"):
        if extra not in enabled:
            enabled.append(extra)
    report = run_evaluation(config=cfg, phases=enabled)
    pipe = next((p for p in report.phases if p.phase_id == "pipeline"), None)
    assert pipe is not None
    assert_phase_ok(pipe, allow_skip=True)

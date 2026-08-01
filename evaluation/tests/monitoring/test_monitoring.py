import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
def test_monitoring_phase():
    report = run_feature("monitoring", skip_llm=True, skip_heavy_ml=True)
    assert_phase_ok(report, allow_skip=True)

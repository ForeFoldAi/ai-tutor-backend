import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.llm
@pytest.mark.critical
def test_ai_tutor_phase():
    report = run_feature("ai_tutor", skip_heavy_ml=False, skip_llm=False)
    assert_phase_ok(report, allow_skip=True)

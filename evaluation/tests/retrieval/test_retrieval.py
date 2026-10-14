import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.retrieval
@pytest.mark.critical
def test_retrieval_phase():
    report = run_feature("retrieval", skip_heavy_ml=False, skip_llm=True)
    assert_phase_ok(report, allow_skip=True)

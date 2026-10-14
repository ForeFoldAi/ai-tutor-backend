import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
def test_topics_phase():
    report = run_feature("topics", skip_heavy_ml=False)
    assert_phase_ok(report, allow_skip=True)

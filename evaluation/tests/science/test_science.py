import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
def test_science_experiments_phase():
    report = run_feature("science_experiments")
    assert_phase_ok(report, allow_skip=True)

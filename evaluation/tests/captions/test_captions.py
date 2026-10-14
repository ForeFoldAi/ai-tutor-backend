import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
def test_captions_phase():
    report = run_feature("captions", skip_heavy_ml=False)
    assert_phase_ok(report, allow_skip=True)

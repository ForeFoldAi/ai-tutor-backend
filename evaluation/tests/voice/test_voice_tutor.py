import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
def test_voice_tutor_phase():
    report = run_feature("voice_tutor")
    assert_phase_ok(report, allow_skip=True)

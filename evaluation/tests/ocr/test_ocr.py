import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.heavy_ml
def test_ocr_phase():
    report = run_feature("ocr", skip_heavy_ml=False)
    assert_phase_ok(report, allow_skip=True)

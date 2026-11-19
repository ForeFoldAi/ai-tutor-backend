import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.images
@pytest.mark.heavy_ml
@pytest.mark.critical
def test_images_phase():
    report = run_feature("images", skip_heavy_ml=False, skip_llm=True)
    assert_phase_ok(report, allow_skip=True)

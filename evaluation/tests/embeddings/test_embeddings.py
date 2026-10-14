import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.critical
def test_embeddings_phase():
    report = run_feature("embeddings", skip_heavy_ml=False)
    assert_phase_ok(report, allow_skip=True)

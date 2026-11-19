import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.heavy_ml
def test_tables_phase():
    report = run_feature("tables", skip_heavy_ml=False)
    assert_phase_ok(report, allow_skip=True)

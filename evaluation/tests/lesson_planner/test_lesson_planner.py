import pytest

from evaluation.tests.conftest import assert_phase_ok, run_feature


@pytest.mark.eval
@pytest.mark.llm
def test_lesson_planner_phase():
    report = run_feature("lesson_planner", skip_llm=False)
    assert_phase_ok(report, allow_skip=True)

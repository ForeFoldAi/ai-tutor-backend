from app.modules.teacher.lesson_planner.constants import ArtifactType
from app.services.lesson_planner.agents.base import run_artifact_agent
from app.services.lesson_planner.state import PlannerState

_ARTIFACT = ArtifactType.LESSON_PLAN.value


def generate(state: PlannerState) -> dict:
    return run_artifact_agent(state, _ARTIFACT)

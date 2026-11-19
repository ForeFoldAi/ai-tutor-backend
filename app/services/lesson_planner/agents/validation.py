from app.services.lesson_planner.schemas import ARTIFACT_SCHEMAS
from app.services.lesson_planner.state import PlannerState

_MARKDOWN_ARTIFACTS = frozenset(
    {"lesson_plan", "teaching_notes", "examples", "worksheet", "quiz", "homework", "ppt_outline"}
)


def validate(state: PlannerState) -> list[str]:
    errors: list[str] = []
    outputs = state.get("outputs") or {}
    for artifact_type, data in outputs.items():
        if artifact_type in _MARKDOWN_ARTIFACTS and isinstance(data, dict) and data.get("format") == "markdown":
            md = data.get("markdown")
            if not isinstance(md, str) or len(md.strip()) < 100:
                errors.append(f"{artifact_type}: markdown content too short or missing")
            continue
        schema = ARTIFACT_SCHEMAS.get(artifact_type)
        if not schema:
            continue
        try:
            schema.model_validate(data)
        except Exception as exc:
            errors.append(f"{artifact_type}: {exc}")
    return errors

"""ponytail: topic focus helper must stay in sync with generate payload field `topics`."""

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import initial_state_from_payload


def _check() -> None:
    empty = topics_focus_block({})
    assert empty == "", empty

    state = initial_state_from_payload(
        {
            "grade": "CLASS_9",
            "subject": "Mathematics",
            "chapter_name": "Fractions",
            "topics": ["Mixed Numbers", "Improper Fractions"],
            "requested_artifacts": ["lesson_plan", "quiz"],
        },
        job_id="1",
        user_id="2",
        lesson_plan_id="3",
    )
    assert state["selected_topics"] == ["Mixed Numbers", "Improper Fractions"]
    block = topics_focus_block(state)
    assert "Mixed Numbers" in block and "Improper Fractions" in block
    assert "ONLY" in block


if __name__ == "__main__":
    _check()
    print("prompt_focus_check: ok")

from app.services.lesson_planner.checkpoint.store import (
    WORKFLOW_NODES,
    clear_checkpoint,
    load_checkpoint,
    next_node_after,
    save_checkpoint,
)


def test_checkpoint_roundtrip():
    job_id = "test-job-checkpoint"
    state = {"job_id": job_id, "outputs": {"lesson_plan": {"title": "T"}}}
    save_checkpoint(job_id, state=state, last_node="retrieve_context")
    loaded = load_checkpoint(job_id)
    assert loaded is not None
    assert loaded["last_node"] == "retrieve_context"
    assert "retrieve_context" in loaded["completed_nodes"]
    clear_checkpoint(job_id)
    assert load_checkpoint(job_id) is None


def test_next_node_after():
    assert next_node_after(None) == WORKFLOW_NODES[0]
    assert next_node_after("retrieve_context") == "retrieve_figures"
    assert next_node_after("persist") is None

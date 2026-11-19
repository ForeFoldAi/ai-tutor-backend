from __future__ import annotations

import logging
from typing import Any, Callable

from app.modules.teacher.lesson_planner.constants import (
    WS_ARTIFACT_COMPLETED,
    WS_ARTIFACT_STARTED,
    WS_COMPLETED,
    WS_ERROR,
    WS_JOB_STARTED,
    WS_PROGRESS,
)
from app.services.lesson_planner.agents.base import run_artifact_agent
from app.services.lesson_planner.agents.validation import validate
from app.services.lesson_planner.algorithms.pedagogy import enrich_pedagogy_metadata
from app.services.lesson_planner.algorithms.worksheet_balance import balance_worksheet_questions
from app.services.lesson_planner.checkpoint.store import (
    WORKFLOW_NODES,
    clear_checkpoint,
    load_checkpoint,
    next_node_after,
    save_checkpoint,
)
from app.services.lesson_planner.observability.tracing import trace_span
from app.services.lesson_planner.redis.job_state import is_cancelled, publish_event
from app.services.lesson_planner.retrieval.chroma_store import search_question_bank
from app.services.lesson_planner.retrieval.experiments import retrieve_experiments_for_lesson
from app.services.lesson_planner.retrieval.figures import retrieve_figures_for_lesson
from app.services.lesson_planner.retrieval.hybrid import hybrid_retrieve
from app.services.lesson_planner.state import PlannerState

logger = logging.getLogger(__name__)

_NODE_HANDLERS: dict[str, Callable[[PlannerState], PlannerState]] = {}


def _node(name: str):
    def decorator(fn: Callable[[PlannerState], PlannerState]):
        _NODE_HANDLERS[name] = fn
        return fn
    return decorator


@_node("retrieve_context")
def _retrieve_context(state: PlannerState) -> PlannerState:
    with trace_span("lesson_planner.retrieve_context"):
        query = f"{state.get('chapter_name', '')} {state.get('learning_objectives', '')}".strip()
        context, chunks = hybrid_retrieve(
            query or state.get("subject", ""),
            board=state.get("board"),
            grade=state.get("grade", ""),
            subject=state.get("subject", ""),
            chapter_id=state.get("chapter_id"),
        )
        state["chapter_context"] = context
        meta = state.setdefault("metadata", {})
        meta["retrieved_chunks"] = len(chunks)
        bank = search_question_bank(
            query or state.get("chapter_name", ""),
            board=state.get("board"),
            grade=state.get("grade", ""),
            subject=state.get("subject", ""),
            chapter_id=state.get("chapter_id"),
            k=6,
        )
        meta["question_bank_hits"] = len(bank)
        if bank:
            meta["question_bank_sample"] = [b["text"][:200] for b in bank[:3]]
    return state


@_node("retrieve_figures")
def _retrieve_figures(state: PlannerState) -> PlannerState:
    with trace_span("lesson_planner.retrieve_figures"):
        figures = retrieve_figures_for_lesson(
            board=state.get("board"),
            grade=state.get("grade", ""),
            subject=state.get("subject", ""),
            chapter_id=state.get("chapter_id"),
            chapter_name=state.get("chapter_name", ""),
            learning_objectives=state.get("learning_objectives", ""),
        )
        state["figures"] = figures
        state.setdefault("metadata", {})["figure_count"] = len(figures)
    return state


@_node("retrieve_experiments")
def _retrieve_experiments(state: PlannerState) -> PlannerState:
    with trace_span("lesson_planner.retrieve_experiments"):
        experiments = retrieve_experiments_for_lesson(
            subject=state.get("subject", ""),
            grade=state.get("grade", ""),
            chapter_name=state.get("chapter_name", ""),
            learning_objectives=state.get("learning_objectives", ""),
        )
        state["experiments"] = experiments
        state.setdefault("metadata", {})["experiment_count"] = len(experiments)
    return state


@_node("enrich_pedagogy")
def _enrich_pedagogy(state: PlannerState) -> PlannerState:
    with trace_span("lesson_planner.enrich_pedagogy"):
        pedagogy = enrich_pedagogy_metadata(state)
        state.setdefault("metadata", {})["pedagogy"] = pedagogy
    return state


@_node("parallel_generation")
def _parallel_generation(state: PlannerState) -> PlannerState:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    job_id = state.get("job_id", "")
    requested = state.get("requested_artifacts") or []
    outputs: dict[str, dict[str, Any]] = dict(state.get("outputs") or {})
    total = max(len(requested), 1)
    completed = 0

    def _generate_one(artifact_type: str) -> tuple[str, dict[str, Any] | None, str | None]:
        if is_cancelled(job_id):
            return artifact_type, None, "cancelled"
        try:
            publish_event(job_id, {"event": WS_ARTIFACT_STARTED, "artifact": artifact_type})
            data = run_artifact_agent(state, artifact_type)
            return artifact_type, data, None
        except Exception as exc:
            logger.exception("Artifact generation failed: %s", artifact_type)
            return artifact_type, None, str(exc)

    with trace_span("lesson_planner.parallel_generation", attributes={"artifact_count": len(requested)}):
        with ThreadPoolExecutor(max_workers=min(4, len(requested) or 1)) as pool:
            futures = {pool.submit(_generate_one, art): art for art in requested}
            for future in as_completed(futures):
                artifact_type, data, err = future.result()
                completed += 1
                progress = int((completed / total) * 75) + 15
                publish_event(
                    job_id,
                    {
                        "event": WS_PROGRESS,
                        "progress": progress,
                        "message": f"Generating {artifact_type.replace('_', ' ').title()}",
                    },
                )
                if err == "cancelled":
                    state.setdefault("errors", []).append(f"{artifact_type}: cancelled")
                    continue
                if err:
                    state.setdefault("errors", []).append(f"{artifact_type}: {err}")
                    publish_event(job_id, {"event": WS_ERROR, "artifact": artifact_type, "message": err})
                    continue
                if data:
                    outputs[artifact_type] = data
                    publish_event(
                        job_id,
                        {"event": WS_ARTIFACT_COMPLETED, "artifact": artifact_type, "data": data},
                    )

    state["outputs"] = outputs
    return state


def _postprocess_worksheet(state: PlannerState) -> None:
    from app.modules.teacher.lesson_planner.constants import ArtifactType

    ws = (state.get("outputs") or {}).get(ArtifactType.WORKSHEET.value)
    if not isinstance(ws, dict):
        return
    flat: list[dict[str, Any]] = []
    for key in ("fill_blanks", "true_false", "short_answer", "long_answer", "application_questions"):
        for item in ws.get(key) or []:
            if isinstance(item, dict):
                flat.append(item)
    if flat:
        ws["difficulty_balance"] = balance_worksheet_questions(flat, grade=state.get("grade", ""))


@_node("validation")
def _validation(state: PlannerState) -> PlannerState:
    with trace_span("lesson_planner.validation"):
        _postprocess_worksheet(state)
        validation_errors = validate(state)
        if validation_errors:
            state.setdefault("errors", []).extend(validation_errors)
        requested = set(state.get("requested_artifacts") or [])
        outputs = state.get("outputs") or {}
        missing = requested - set(outputs.keys())
        if missing:
            state.setdefault("errors", []).append(f"Missing artifacts: {', '.join(sorted(missing))}")
    return state


@_node("persist")
def _persist_marker(state: PlannerState) -> PlannerState:
    state.setdefault("metadata", {})["persisted"] = True
    return state


def run_planner_workflow(state: PlannerState, *, resume: bool = False) -> PlannerState:
    job_id = state.get("job_id", "")
    checkpoint = load_checkpoint(job_id) if resume else None
    completed: set[str] = set()
    if checkpoint:
        state = dict(checkpoint.get("state") or state)
        completed = set(checkpoint.get("completed_nodes") or [])
        publish_event(job_id, {"event": WS_PROGRESS, "progress": 12, "message": "Resuming generation"})
    else:
        publish_event(job_id, {"event": WS_JOB_STARTED, "job_id": job_id})
        publish_event(job_id, {"event": WS_PROGRESS, "progress": 5, "message": "Retrieving context"})

    start_node = next_node_after(checkpoint.get("last_node") if checkpoint else None) if resume else WORKFLOW_NODES[0]
    if resume and start_node is None:
        start_node = WORKFLOW_NODES[0]

    start_idx = WORKFLOW_NODES.index(start_node) if start_node in WORKFLOW_NODES else 0

    last_run_node = checkpoint.get("last_node") if checkpoint else None
    with trace_span("lesson_planner.workflow", attributes={"job_id": job_id, "resume": resume}):
        try:
            for node_name in WORKFLOW_NODES[start_idx:]:
                if node_name in completed and resume:
                    continue
                if is_cancelled(job_id):
                    save_checkpoint(job_id, state=dict(state), last_node=node_name)
                    break
                publish_event(
                    job_id,
                    {
                        "event": WS_PROGRESS,
                        "progress": 5 + int(15 * (WORKFLOW_NODES.index(node_name) / max(len(WORKFLOW_NODES) - 1, 1))),
                        "message": node_name.replace("_", " ").title(),
                    },
                )
                handler = _NODE_HANDLERS[node_name]
                state = handler(state)
                last_run_node = node_name
                save_checkpoint(job_id, state=dict(state), last_node=node_name)
        except Exception:
            save_checkpoint(job_id, state=dict(state), last_node=last_run_node or WORKFLOW_NODES[0])
            raise

    if is_cancelled(job_id):
        publish_event(job_id, {"event": "cancelled", "job_id": job_id})
    else:
        clear_checkpoint(job_id)
        publish_event(
            job_id,
            {
                "event": WS_COMPLETED,
                "result": {
                    "outputs": state.get("outputs") or {},
                    "metadata": state.get("metadata") or {},
                    "errors": state.get("errors") or [],
                },
            },
        )
        publish_event(job_id, {"event": WS_PROGRESS, "progress": 100, "message": "Completed"})
    return state


# ponytail: LangGraph compile kept for future; sequential runner supports resume
def build_lesson_planner_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PlannerState)
    for name in WORKFLOW_NODES:
        graph.add_node(name, _NODE_HANDLERS[name])
    graph.add_edge(START, WORKFLOW_NODES[0])
    for a, b in zip(WORKFLOW_NODES, WORKFLOW_NODES[1:]):
        graph.add_edge(a, b)
    graph.add_edge(WORKFLOW_NODES[-1], END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_lesson_planner_graph()
    return _compiled_graph

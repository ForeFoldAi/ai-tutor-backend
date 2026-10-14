from __future__ import annotations

import json
import logging
from typing import Any

from app.services.lesson_planner.redis.job_state import get_sync_redis

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "lesson_planner:checkpoint:"
_CHECKPOINT_TTL = 86_400 * 7  # 7 days

WORKFLOW_NODES = (
    "retrieve_context",
    "retrieve_figures",
    "retrieve_experiments",
    "enrich_pedagogy",
    "parallel_generation",
    "validation",
    "persist",
)


def _key(job_id: str) -> str:
    return f"{CHECKPOINT_PREFIX}{job_id}"


def save_checkpoint(job_id: str, *, state: dict[str, Any], last_node: str) -> None:
    r = get_sync_redis()
    payload = {
        "state": state,
        "last_node": last_node,
        "completed_nodes": list(_nodes_through(last_node)),
    }
    r.set(_key(job_id), json.dumps(payload, default=str), ex=_CHECKPOINT_TTL)


def load_checkpoint(job_id: str) -> dict[str, Any] | None:
    r = get_sync_redis()
    raw = r.get(_key(job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def clear_checkpoint(job_id: str) -> None:
    r = get_sync_redis()
    r.delete(_key(job_id))


def _nodes_through(last_node: str) -> list[str]:
    if last_node not in WORKFLOW_NODES:
        return []
    idx = WORKFLOW_NODES.index(last_node)
    return list(WORKFLOW_NODES[: idx + 1])


def next_node_after(last_node: str | None) -> str | None:
    if not last_node:
        return WORKFLOW_NODES[0]
    if last_node not in WORKFLOW_NODES:
        return WORKFLOW_NODES[0]
    idx = WORKFLOW_NODES.index(last_node)
    if idx + 1 >= len(WORKFLOW_NODES):
        return None
    return WORKFLOW_NODES[idx + 1]

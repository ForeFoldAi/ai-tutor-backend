from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.services.lesson_planner.observability.metrics import (
    ARTIFACT_GENERATION,
    GENERATION_JOBS,
    GENERATION_LATENCY,
)

metrics_router = APIRouter(tags=["observability"])


@metrics_router.get("/health/lesson-planner")
def lesson_planner_health() -> dict:
    from app.services.lesson_planner.observability.tracing import setup_lesson_planner_otel

    return {
        "ok": True,
        "otel_enabled": setup_lesson_planner_otel(),
        "pydantic_ai": _pydantic_ai_status(),
    }


def _pydantic_ai_status() -> bool:
    try:
        from app.services.lesson_planner.pydantic_ai_agent import _pydantic_ai_available
        return _pydantic_ai_available()
    except Exception:
        return False


@metrics_router.get("/metrics/lesson-planner")
def lesson_planner_metrics() -> Response:
    """Prometheus metrics for lesson planner generation."""
    # ponytail: touch counters so they appear in registry even before first job
    _ = (GENERATION_JOBS, GENERATION_LATENCY, ARTIFACT_GENERATION)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

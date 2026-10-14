from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_tracer = None
_initialized = False


def setup_lesson_planner_otel() -> bool:
    """Configure OpenTelemetry tracer for lesson planner (idempotent)."""
    global _tracer, _initialized
    if _initialized:
        return _tracer is not None

    _initialized = True
    if os.environ.get("LESSON_PLANNER_OTEL_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        resource = Resource.create({"service.name": "ai-tutor-lesson-planner"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("app.services.lesson_planner")
        logger.info("Lesson planner OpenTelemetry enabled → %s", endpoint)
        return True
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed (non-fatal): %s", exc)
        return False


def get_tracer():
    setup_lesson_planner_otel()
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        return trace.get_tracer("app.services.lesson_planner")
    except Exception:
        return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
    tracer = get_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, str(value))
        yield

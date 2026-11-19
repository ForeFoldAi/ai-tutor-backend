"""Optional OpenTelemetry helpers for monitoring (extends lesson-planner pattern)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_monitoring_otel() -> bool:
    if os.environ.get("MONITORING_OTEL_ENABLED", "false").lower() not in ("1", "true", "yes"):
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
        provider = TracerProvider(resource=Resource.create({"service.name": "learnmestro-api"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(provider)
        logger.info("Monitoring OpenTelemetry enabled → %s", endpoint)
        return True
    except Exception as exc:
        logger.warning("Monitoring OTel setup failed (non-fatal): %s", exc)
        return False

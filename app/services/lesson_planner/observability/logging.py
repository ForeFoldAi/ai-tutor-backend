from __future__ import annotations

import logging
import uuid

import structlog

logger = logging.getLogger(__name__)


def configure_structlog() -> None:
    try:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )
    except Exception:
        # ponytail: structlog optional at import time
        pass


def bind_request_id(request_id: str | None = None) -> None:
    rid = request_id or str(uuid.uuid4())
    try:
        structlog.contextvars.bind_contextvars(request_id=rid)
    except Exception:
        pass

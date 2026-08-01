from __future__ import annotations

import os
import sys

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings
from app.modules.teacher.lesson_planner.constants import (
    QUEUE_AUTOSAVE,
    QUEUE_EXPORT,
    QUEUE_GENERATE,
    QUEUE_REGENERATE,
)
from app.modules.users.models import User  # noqa: F401 — FK target for lesson planner tables
from app.modules.teacher.lesson_planner.models import (  # noqa: F401
    LessonArtifact,
    LessonExport,
    LessonJob,
    LessonPlan,
    LessonVersion,
)

settings = get_settings()

# ponytail: macOS prefork loads PyTorch/BGE in forked children → SIGABRT; solo avoids fork.
_worker_pool = os.environ.get("CELERY_WORKER_POOL") or ("solo" if sys.platform == "darwin" else "prefork")

from app.modules.learning_intelligence.constants import QUEUE_LIA

QUEUE_MAIL = "mail"

celery_app = Celery(
    "lesson_planner",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.services.lesson_planner.workers.tasks",
        "app.services.mail.tasks",
        "app.services.learning_intelligence.workers.tasks",
    ],
)

celery_app.conf.update(
    worker_pool=_worker_pool,
    worker_concurrency=1 if _worker_pool == "solo" else int(os.environ.get("CELERY_WORKER_CONCURRENCY", "2")),
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.mail_reminder_tz,
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue=QUEUE_GENERATE,
    task_queues={
        QUEUE_GENERATE: {"exchange": QUEUE_GENERATE, "routing_key": QUEUE_GENERATE},
        QUEUE_EXPORT: {"exchange": QUEUE_EXPORT, "routing_key": QUEUE_EXPORT},
        QUEUE_AUTOSAVE: {"exchange": QUEUE_AUTOSAVE, "routing_key": QUEUE_AUTOSAVE},
        QUEUE_REGENERATE: {"exchange": QUEUE_REGENERATE, "routing_key": QUEUE_REGENERATE},
        QUEUE_MAIL: {"exchange": QUEUE_MAIL, "routing_key": QUEUE_MAIL},
        QUEUE_LIA: {"exchange": QUEUE_LIA, "routing_key": QUEUE_LIA},
    },
    task_routes={
        "lesson_planner.generate": {"queue": QUEUE_GENERATE},
        "lesson_planner.export": {"queue": QUEUE_EXPORT},
        "lesson_planner.autosave": {"queue": QUEUE_AUTOSAVE},
        "lesson_planner.regenerate": {"queue": QUEUE_REGENERATE},
        "mail.send_session_reminders": {"queue": QUEUE_MAIL},
        "mail.send_assignment_reminders": {"queue": QUEUE_MAIL},
        "lia.refresh_student_twin": {"queue": QUEUE_LIA},
        "lia.weekly_summaries": {"queue": QUEUE_LIA},
        "lia.monthly_summaries": {"queue": QUEUE_LIA},
        "lia.run_prediction": {"queue": QUEUE_LIA},
        "lia.seed_concept_graph": {"queue": QUEUE_LIA},
    },
    beat_schedule={
        "mail-session-reminders": {
            "task": "mail.send_session_reminders",
            "schedule": crontab(
                hour=settings.mail_session_reminder_hour,
                minute=settings.mail_reminder_minute,
            ),
        },
        "mail-assignment-reminders": {
            "task": "mail.send_assignment_reminders",
            "schedule": crontab(
                hour=settings.mail_assignment_reminder_hour,
                minute=settings.mail_reminder_minute,
            ),
        },
        "lia-weekly-summaries": {
            "task": "lia.weekly_summaries",
            "schedule": crontab(hour=6, minute=0, day_of_week=0),
        },
        "lia-monthly-summaries": {
            "task": "lia.monthly_summaries",
            "schedule": crontab(hour=6, minute=30, day_of_month=1),
        },
    },
    task_annotations={
        "*": {
            "rate_limit": os.environ.get("LESSON_PLANNER_TASK_RATE_LIMIT", "30/m"),
        }
    },
    broker_transport_options={
        "visibility_timeout": int(os.environ.get("LESSON_PLANNER_VISIBILITY_TIMEOUT", "3600")),
    },
    task_reject_on_worker_lost=True,
    task_time_limit=int(os.environ.get("LESSON_PLANNER_TASK_TIME_LIMIT", "1800")),
    task_soft_time_limit=int(os.environ.get("LESSON_PLANNER_TASK_SOFT_TIME_LIMIT", "1700")),
)

# Dead-letter: failed tasks after max retries land in Redis result with FAILURE state.
celery_app.conf.task_default_retry_delay = 30
celery_app.conf.task_max_retries = int(os.environ.get("LESSON_PLANNER_MAX_RETRIES", "3"))

"""LIA Celery background workers."""

from __future__ import annotations

import logging

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.learning_intelligence.constants import PERIOD_MONTHLY, PERIOD_WEEKLY, QUEUE_LIA
from app.modules.learning_intelligence import service as lia_svc
from app.services.learning_intelligence.agents.memory import build_period_summary
from app.services.learning_intelligence.agents.student_modeling import update_twin_from_events
from app.services.learning_intelligence.algorithms.concept_graph import seed_concept_graph

logger = logging.getLogger(__name__)


@celery_app.task(name="lia.refresh_student_twin", queue=QUEUE_LIA)
def refresh_student_twin(student_user_id: int) -> dict:
    db = SessionLocal()
    try:
        profile = update_twin_from_events(db, student_user_id)
        db.commit()
        return {"student_user_id": student_user_id, "twin_version": profile.twin_version}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(name="lia.seed_concept_graph", queue=QUEUE_LIA)
def seed_concept_graph_task(
    board: str | None = None,
    class_level: str | None = None,
    subject_name: str | None = None,
) -> dict:
    db = SessionLocal()
    try:
        result = seed_concept_graph(db, board=board, class_level=class_level, subject_name=subject_name)
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(name="lia.weekly_summaries", queue=QUEUE_LIA)
def weekly_summaries() -> int:
    from sqlalchemy import select
    from app.modules.learning_intelligence.models import LiaLearningEvent

    db = SessionLocal()
    count = 0
    try:
        student_ids = db.scalars(
            select(LiaLearningEvent.student_user_id).distinct().limit(500)
        ).all()
        for sid in student_ids:
            update_twin_from_events(db, sid)
            build_period_summary(db, sid, PERIOD_WEEKLY)
            count += 1
        db.commit()
        return count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(name="lia.monthly_summaries", queue=QUEUE_LIA)
def monthly_summaries() -> int:
    from sqlalchemy import select
    from app.modules.learning_intelligence.models import LiaLearningEvent

    db = SessionLocal()
    count = 0
    try:
        student_ids = db.scalars(
            select(LiaLearningEvent.student_user_id).distinct().limit(500)
        ).all()
        for sid in student_ids:
            update_twin_from_events(db, sid)
            build_period_summary(db, sid, PERIOD_MONTHLY)
            count += 1
        db.commit()
        return count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(name="lia.run_prediction", queue=QUEUE_LIA)
def run_prediction_task(student_user_id: int, concept_key: str | None = None) -> dict:
    db = SessionLocal()
    try:
        result = lia_svc.run_prediction(db, student_user_id, concept_key)
        db.commit()
        return result.model_dump()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

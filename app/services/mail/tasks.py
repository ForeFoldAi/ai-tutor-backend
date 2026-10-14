"""Celery tasks for daily mail reminders."""

from __future__ import annotations

import logging

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.services.mail.reminders import run_assignment_reminders, run_session_reminders

logger = logging.getLogger(__name__)


@celery_app.task(name="mail.send_session_reminders", queue="mail")
def send_session_reminders() -> dict:
    db = SessionLocal()
    try:
        result = run_session_reminders(db)
        db.commit()
        return result
    except Exception:
        db.rollback()
        logger.exception("mail.send_session_reminders failed")
        raise
    finally:
        db.close()


@celery_app.task(name="mail.send_assignment_reminders", queue="mail")
def send_assignment_reminders() -> dict:
    db = SessionLocal()
    try:
        result = run_assignment_reminders(db)
        db.commit()
        return result
    except Exception:
        db.rollback()
        logger.exception("mail.send_assignment_reminders failed")
        raise
    finally:
        db.close()

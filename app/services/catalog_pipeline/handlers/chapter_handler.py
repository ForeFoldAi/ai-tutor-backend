"""Resilient chapter processing handler for confirmed textbook chapters."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from app.core.database import SessionLocal
from app.modules.catalog.models import (
    ChapterStatusEnum,
    Textbook,
    TextbookChapter,
    TextbookStatusEnum,
)

logger = logging.getLogger(__name__)


def fail_interrupted_jobs(*, include_queued: bool) -> int:
    """Mark jobs left running by a dead process as FAILED so Retry is offered.

    Call only from the process that runs catalog jobs, at its startup, and only when
    CATALOG_RECOVER_ON_STARTUP=true: any other process sharing the database (a second worker, a
    developer laptop pointed at the same DB) would have its live jobs failed. include_queued also
    fails CONFIRMED/ANALYZING books, whose in-memory BackgroundTasks died with the process; a
    Celery worker must not, since those tasks are still in the queue.
    """
    if os.environ.get("CATALOG_RECOVER_ON_STARTUP", "false").lower() not in ("1", "true", "yes"):
        return 0
    book_states = [TextbookStatusEnum.PROCESSING]
    if include_queued:
        book_states += [TextbookStatusEnum.CONFIRMED, TextbookStatusEnum.ANALYZING]
    db = SessionLocal()
    try:
        n = db.execute(
            update(Textbook).where(Textbook.status.in_(book_states)).values(status=TextbookStatusEnum.FAILED)
        ).rowcount
        db.execute(
            update(TextbookChapter)
            .where(TextbookChapter.status == ChapterStatusEnum.PROCESSING)
            .values(status=ChapterStatusEnum.FAILED, error_message="Interrupted by a server restart; use Retry.")
        )
        db.commit()
        if n:
            logger.warning("Marked %d interrupted textbook job(s) as FAILED", n)
        return n
    finally:
        db.close()


def handle_textbook_chapters(textbook_id: int) -> dict[str, Any]:
    """
    Process all confirmed chapters of a textbook sequentially.
    Slices each chapter, chunks, embeds, extracts images, and updates database records.
    """
    db = SessionLocal()
    try:
        tb = db.get(Textbook, textbook_id)
        if not tb:
            logger.warning("Textbook %s not found; skipping chapter processing", textbook_id)
            return {"status": "skipped", "reason": "not_found"}
        # Atomic claim: a duplicate dispatch (double click, redelivered task) finds the book already PROCESSING.
        # updated_at doubles as this run's token: Stop or a newer claim changes it and this run exits.
        run_token = datetime.now(UTC)
        claimed = db.execute(
            update(Textbook)
            .where(Textbook.id == textbook_id, Textbook.status == TextbookStatusEnum.CONFIRMED)
            .values(status=TextbookStatusEnum.PROCESSING, updated_at=run_token)
        ).rowcount
        db.commit()
        if not claimed:
            logger.info("Textbook %s is %s (stopped or already running); skipping", textbook_id, tb.status)
            return {"status": "skipped", "reason": "not_confirmed"}
        db.refresh(tb)

        # Get all valid chapters that are not non-chapter sections
        chapters = list(
            db.scalars(
                select(TextbookChapter)
                .where(
                    TextbookChapter.textbook_id == textbook_id,
                    TextbookChapter.is_non_chapter_section == False,
                )
                .order_by(TextbookChapter.start_pdf_page)
            )
        )

        from app.services.textbook_segmentation import process_textbook_chapter_background

        all_ok = True
        completed_count = 0
        failed_count = 0

        for ch in chapters:
            # ponytail: cooperative stop — the admin "Stop" flips the textbook to FAILED and we
            # exit before the next chapter; the chapter already running finishes first.
            db.refresh(tb)
            if tb.status != TextbookStatusEnum.PROCESSING or tb.updated_at != run_token:
                logger.info("Textbook %s processing stopped by admin or taken over by a newer run", textbook_id)
                all_ok = False
                break
            db.refresh(ch)
            if ch.status == ChapterStatusEnum.COMPLETED:
                completed_count += 1
                continue
            try:
                logger.info("Processing chapter %s (Ch #%s: %s) for textbook %s", ch.id, ch.chapter_number, ch.chapter_title, textbook_id)
                process_textbook_chapter_background(ch.id)
                db.refresh(ch)
                if ch.status == ChapterStatusEnum.FAILED:
                    all_ok = False
                    failed_count += 1
                else:
                    completed_count += 1
            except Exception as ch_err:
                logger.exception("Chapter %s processing error: %s", ch.id, ch_err)
                all_ok = False
                failed_count += 1

        # Only the run that still owns the book writes the outcome (Stop already set FAILED).
        db.execute(
            update(Textbook)
            .where(
                Textbook.id == textbook_id,
                Textbook.status == TextbookStatusEnum.PROCESSING,
                Textbook.updated_at == run_token,
            )
            .values(status=TextbookStatusEnum.COMPLETED if all_ok else TextbookStatusEnum.FAILED)
        )
        db.commit()
        logger.info(
            "Textbook %s chapter processing finished: %d completed, %d failed",
            textbook_id,
            completed_count,
            failed_count,
        )
        return {
            "status": "completed" if all_ok else "partial_failure",
            "textbook_id": textbook_id,
            "completed": completed_count,
            "failed": failed_count,
        }

    except Exception as exc:
        logger.exception("Textbook %s processing failed: %s", textbook_id, exc)
        try:
            tb = db.get(Textbook, textbook_id)
            if tb:
                tb.status = TextbookStatusEnum.FAILED
                db.commit()
        except Exception as db_err:
            logger.error("Failed to mark textbook %s as FAILED: %s", textbook_id, db_err)
        raise exc
    finally:
        db.close()

"""Resilient chapter processing handler for confirmed textbook chapters."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from app.core.database import SessionLocal
from app.modules.catalog.models import (
    ChapterStatusEnum,
    Textbook,
    TextbookChapter,
    TextbookStatusEnum,
)

logger = logging.getLogger(__name__)


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

        tb.status = TextbookStatusEnum.PROCESSING
        db.commit()

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

        tb.status = TextbookStatusEnum.COMPLETED if all_ok else TextbookStatusEnum.FAILED
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

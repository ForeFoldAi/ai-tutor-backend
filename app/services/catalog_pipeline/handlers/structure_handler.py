"""Resilient structure analysis handler for full textbook PDF segmentation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.core.database import SessionLocal
from app.modules.catalog.models import (
    ChapterStatusEnum,
    Textbook,
    TextbookChapter,
    TextbookStatusEnum,
)

logger = logging.getLogger(__name__)


def _local_path_for_upload(file_path: str | None) -> str:
    from app.services.image_service.storage_backend import materialize_textbook_file

    return materialize_textbook_file(file_path or "")


def handle_textbook_analysis(textbook_id: int) -> dict[str, Any]:
    """
    Run hybrid chapter detection on uploaded textbook.
    Detects TOC, front matter, headings, and chapter boundaries.
    """
    db = SessionLocal()
    try:
        tb = db.get(Textbook, textbook_id)
        if not tb or not tb.file_path:
            logger.warning("Textbook %s or file_path missing; skipping analysis", textbook_id)
            return {"status": "skipped", "reason": "not_found"}

        tb.status = TextbookStatusEnum.ANALYZING
        db.commit()

        local_path = _local_path_for_upload(tb.file_path)
        if not local_path or not os.path.isfile(local_path):
            raise FileNotFoundError(f"Source file missing for textbook {textbook_id}: {tb.file_path}")

        from app.services.textbook_structure import detect_textbook_structure

        res = detect_textbook_structure(local_path)
        tb.total_pages = res.get("total_pages", 0)
        tb.pdf_type = res.get("pdf_type", "text_based")
        tb.page_offset = res.get("page_offset", 0)
        tb.confidence_score = res.get("confidence_score", 1.0)
        tb.detection_method = res.get("detection_method", "universal_hybrid")
        tb.detection_summary = json.dumps({
            "page_offset": res.get("page_offset", 0),
            "confidence_score": res.get("confidence_score", 1.0),
            "detection_method": res.get("detection_method", "universal_hybrid"),
            "needs_manual_review": res.get("needs_manual_review", False),
            "review_warnings": res.get("review_warnings", []),
            "toc_found": res.get("toc_found", False),
            "chapters_detected": len(res.get("chapters", [])),
        })

        # Clear any prior chapters if retried (directly uploaded chapters belong to the publisher, keep them)
        for ch in list(tb.chapters):
            if ch.detection_method != "manual_upload":
                db.delete(ch)
        db.commit()

        # Add detected chapters (with hierarchy)
        for ch_data in res.get("chapters", []):
            flags_json = json.dumps(ch_data.get("confidence_flags", []))
            row = TextbookChapter(
                textbook_id=tb.id,
                chapter_number=ch_data["chapter_number"],
                chapter_title=ch_data["chapter_title"],
                hierarchy_level=ch_data.get("hierarchy_level", "chapter"),
                start_pdf_page=ch_data["start_pdf_page"],
                end_pdf_page=ch_data["end_pdf_page"],
                printed_start_page=ch_data.get("printed_start_page"),
                printed_end_page=ch_data.get("printed_end_page"),
                is_non_chapter_section=False,
                section_type=ch_data.get("section_type", "chapter"),
                detection_method=ch_data.get("detection_method", "toc_body_match"),
                confidence_score=ch_data.get("confidence_score", 1.0),
                confidence_flags=flags_json,
                status=ChapterStatusEnum.DETECTED,
            )
            db.add(row)
            db.commit()
            db.refresh(row)

            # Add sub-chapters / readings if present
            for sub_data in ch_data.get("sub_chapters", []):
                sub_row = TextbookChapter(
                    textbook_id=tb.id,
                    parent_id=row.id,
                    chapter_number=sub_data["chapter_number"],
                    chapter_title=sub_data["chapter_title"],
                    hierarchy_level=sub_data.get("hierarchy_level", "reading"),
                    start_pdf_page=sub_data["start_pdf_page"],
                    end_pdf_page=sub_data["end_pdf_page"],
                    printed_start_page=sub_data.get("printed_start_page"),
                    printed_end_page=sub_data.get("printed_end_page"),
                    is_non_chapter_section=False,
                    section_type="reading",
                    detection_method=sub_data.get("detection_method", "toc_sub_reading"),
                    confidence_score=sub_data.get("confidence_score", 0.90),
                    confidence_flags=json.dumps(sub_data.get("confidence_flags", [])),
                    status=ChapterStatusEnum.DETECTED,
                )
                db.add(sub_row)
            db.commit()

        # Add non-chapter sections (preface, toc, appendix)
        for sec_data in res.get("non_chapter_sections", []):
            row = TextbookChapter(
                textbook_id=tb.id,
                chapter_number=sec_data["chapter_number"],
                chapter_title=sec_data["chapter_title"],
                hierarchy_level=sec_data.get("hierarchy_level", "front_matter"),
                start_pdf_page=sec_data["start_pdf_page"],
                end_pdf_page=sec_data["end_pdf_page"],
                printed_start_page=sec_data.get("printed_start_page"),
                printed_end_page=sec_data.get("printed_end_page"),
                is_non_chapter_section=True,
                section_type=sec_data.get("section_type", "preface"),
                detection_method=sec_data.get("detection_method", "boundary_residual"),
                confidence_score=sec_data.get("confidence_score", 1.0),
                confidence_flags="[]",
                status=ChapterStatusEnum.DETECTED,
            )
            db.add(row)

        tb.status = TextbookStatusEnum.AWAITING_REVIEW
        db.commit()
        detected_count = len(res.get("chapters", []))
        logger.info("Textbook %s analyzed successfully: %d chapters detected", tb.id, detected_count)
        return {
            "status": "completed",
            "textbook_id": textbook_id,
            "chapters_detected": detected_count,
        }

    except Exception as exc:
        logger.exception("Failed to analyze textbook %s: %s", textbook_id, exc)
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

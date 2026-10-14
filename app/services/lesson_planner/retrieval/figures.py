from __future__ import annotations

import logging
from typing import Any

from app.services.image_service.textbook_image_retrieval import related_images_payload
from app.services.lesson_planner.retrieval.hybrid import _collection_name
from app.services.vector_service import retrieve_from_collection

logger = logging.getLogger(__name__)


def retrieve_figures_for_lesson(
    *,
    board: str | None,
    grade: str,
    subject: str,
    chapter_id: str | None,
    chapter_name: str,
    learning_objectives: str,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """Rank textbook figures via image_service pedagogy ranker."""
    if not chapter_id:
        return []

    query = f"{chapter_name} {learning_objectives}".strip()
    collection = _collection_name(board, grade, subject)
    chapter_ids = [chapter_id]

    try:
        docs = retrieve_from_collection(
            query or chapter_name,
            collection_name=collection,
            chapter_ids=chapter_ids,
            k=5,
        )
        figures = related_images_payload(
            chapter_ids,
            [chapter_name],
            chapter_name,
            query or chapter_name,
            docs,
            top_n=top_n,
            # ponytail: never block lesson generation on PDF ML / BLIP extract
            ensure_extract=False,
        )
        return [
            {
                "file_name": f.get("file_name"),
                "caption": f.get("caption"),
                "image_url": f.get("image_url") or f.get("url"),
                "textbook_upload_id": f.get("textbook_upload_id"),
                "page": f.get("page"),
                "image_type": f.get("image_type"),
            }
            for f in (figures or [])
        ]
    except Exception as exc:
        logger.warning("Figure retrieval failed: %s", exc)
        return []

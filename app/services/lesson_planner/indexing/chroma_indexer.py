from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document

from app.services.lesson_planner.retrieval.chroma_store import (
    COLLECTION_QUESTION_BANK,
    COLLECTION_TEXTBOOK_EXPERIMENTS,
    COLLECTION_TEXTBOOK_IMAGES,
    resolve_textbook_collection,
)
from app.services.science_experiment.experiment_catalog import match_science_experiment
from app.services.vector_service import add_documents_to_store, fetch_chapter_chunks

logger = logging.getLogger(__name__)

_QUESTION_PATTERNS = re.compile(
    r"\b(question|exercise|problem|\d+\.|mcq|fill in the blank|true or false)\b",
    re.I,
)


def _scoped_name(base: str, board: str | None, grade: str, subject: str) -> str:
    return f"{base}_{resolve_textbook_collection(board, grade, subject)}"


def index_lesson_planner_collections(
    *,
    board: str | None,
    grade: str,
    subject: str,
    chapter_id: str,
    chapter_name: str,
) -> dict[str, int]:
    """Populate dedicated Chroma collections from existing textbook chunks."""
    source = resolve_textbook_collection(board, grade, subject)
    chunks = fetch_chapter_chunks(source, [chapter_id], limit=500)
    if not chunks:
        logger.info("No chunks to index for chapter %s", chapter_id)
        return {}

    stats: dict[str, int] = {}

    # Question bank: filter chunk text that looks like assessment content
    question_docs = [
        Document(page_content=d.page_content, metadata={**(d.metadata or {}), "kind": "question"})
        for d in chunks
        if _QUESTION_PATTERNS.search(d.page_content or "")
    ]
    if question_docs:
        stats["question_bank"] = add_documents_to_store(
            question_docs,
            collection_name=_scoped_name(COLLECTION_QUESTION_BANK, board, grade, subject),
        )

    # Image captions from metadata (ponytail: index caption text; images stay in PG)
    image_docs: list[Document] = []
    for d in chunks:
        meta = d.metadata or {}
        caption = meta.get("figure_caption") or meta.get("caption")
        if caption:
            image_docs.append(
                Document(
                    page_content=str(caption),
                    metadata={**meta, "kind": "image", "textbook_upload_id": chapter_id},
                )
            )
    if image_docs:
        stats["textbook_images"] = add_documents_to_store(
            image_docs,
            collection_name=_scoped_name(COLLECTION_TEXTBOOK_IMAGES, board, grade, subject),
        )

    # Science experiments catalog entry for chapter topic
    if _is_science(subject):
        experiment = match_science_experiment(chapter_name, grade)
        if experiment:
            exp_doc = Document(
                page_content=f"{experiment.get('conceptName', '')} {experiment.get('learningObjective', '')}",
                metadata={
                    "kind": "experiment",
                    "textbook_upload_id": chapter_id,
                    "experiment_type": (experiment.get("experiment") or {}).get("experimentType"),
                },
            )
            stats["textbook_experiments"] = add_documents_to_store(
                [exp_doc],
                collection_name=_scoped_name(COLLECTION_TEXTBOOK_EXPERIMENTS, board, grade, subject),
            )

    logger.info("Indexed lesson planner collections for %s: %s", chapter_id, stats)
    return stats


def _is_science(subject: str) -> bool:
    s = (subject or "").lower()
    return any(k in s for k in ("science", "physics", "chemistry", "biology"))

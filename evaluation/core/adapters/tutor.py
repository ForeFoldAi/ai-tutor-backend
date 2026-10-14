"""AI Tutor chat adapter."""

from __future__ import annotations

import asyncio
from typing import Any


async def _ask_async(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str],
    board: str,
    class_level: str,
    subject_name: str,
) -> tuple[str, list[dict], Any, Any]:
    from app.services.chat_service import chapter_aware_qa

    return await chapter_aware_qa(
        query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )


def ask_tutor(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str],
    board: str,
    class_level: str,
    subject_name: str,
) -> dict[str, Any]:
    answer, images, math_lesson, science_exp = asyncio.run(
        _ask_async(
            query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            board=board,
            class_level=class_level,
            subject_name=subject_name,
        )
    )
    return {
        "answer": answer or "",
        "images": images or [],
        "math_lesson": math_lesson,
        "science_experiment": science_exp,
    }

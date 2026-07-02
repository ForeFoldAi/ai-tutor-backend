"""
Chapter Awareness Mode — integration test against CBSE Class 9 Social textbook.

Run from ai-tutor-backend:
  .venv/bin/python scripts/test_chapter_awareness_social.py
"""

from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

UPLOAD_ID = "97ce23be-1c71-45f2-942c-34ea82cc205c"
COLLECTION = "CBSE_CLASS_9_Social"
CHAPTER = "Chapter 2 - Understanding the Weather"

CASES = [
    {
        "query": "what is desert?",
        "expect": "awareness",
        "note": "topic from Chapter 1 while studying Chapter 2",
    },
    {
        "query": "what is humidity?",
        "expect": "answer",
        "note": "in-chapter topic",
    },
    {
        "query": "what is democracy?",
        "expect": "awareness",
        "note": "not covered in weather chapter",
    },
]

SEP = "=" * 72


def _is_awareness(text: str) -> bool:
    return "How would you like to continue?" in (text or "")


async def main() -> None:
    from app.services.chapter_scope import (
        ChapterCoverageLevel,
        assess_chapter_coverage,
    )
    from app.services.chat_service import chapter_aware_qa
    from app.services.section_retrieval import retrieve_for_tutor_query

    kw = dict(
        collection_name=COLLECTION,
        chapter_ids=[UPLOAD_ID],
        class_level="CLASS_9",
        board="CBSE",
        subject_name="Social",
        chapter=CHAPTER,
        chapter_names=[CHAPTER],
    )

    print(SEP)
    print("CHAPTER AWARENESS — CBSE CLASS 9 SOCIAL (Chapter 2 Weather)")
    print(SEP)

    passed = 0
    for case in CASES:
        q = case["query"]
        print(f"\nQ: {q!r}  ({case['note']})")

        docs, _, _ = retrieve_for_tutor_query(
            q, collection_name=COLLECTION, chapter_ids=[UPLOAD_ID]
        )
        assessment = assess_chapter_coverage(
            q,
            docs=docs,
            collection_name=COLLECTION,
            chapter_ids=[UPLOAD_ID],
            chapter_names=[CHAPTER],
            board="CBSE",
            class_level="CLASS_9",
            subject_name="Social",
        )
        print(f"  coverage: {assessment.level.value}  topic={assessment.topic_label!r}")

        answer, _, _ = await chapter_aware_qa(q, **kw)
        preview = (answer or "")[:280].replace("\n", " ")
        print(f"  response: {preview}...")

        ok = False
        if case["expect"] == "awareness":
            ok = _is_awareness(answer)
            if assessment.level != ChapterCoverageLevel.NONE:
                print(f"  WARN: expected coverage=none, got {assessment.level.value}")
        else:
            ok = not _is_awareness(answer) and len((answer or "").split()) > 20
            if assessment.level == ChapterCoverageLevel.NONE:
                print("  WARN: expected in-chapter coverage")

        print(f"  CASE: {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1

    print(f"\n{SEP}\n{passed}/{len(CASES)} passed\n{SEP}")
    if passed < len(CASES):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

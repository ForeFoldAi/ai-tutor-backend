"""
End-to-end retrieval test for the weather chapter.

Expected:
  Q1: "what is weather?"              → Fig 2.2  (p1_5.jpg)
  Q2: "what is Precipitation with Rain gauge image?" → Fig 2.6  (rain gauge — may be missing)

Run:
  DATABASE_URL="..." .venv/bin/python3 scripts/test_retrieval_weather.py
"""

from __future__ import annotations

import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

UPLOAD_ID = "e1a8850c-03e9-4456-a6ca-d398c39ae259"
CHAPTER_IDS = [UPLOAD_ID]
COLLECTION = "CBSE_CLASS_9_Social"

QUERIES = [
    ("what is weather?",                                "Fig 2.2  (p1_5.jpg)"),
    ("what is Precipitation with Rain gauge image?",    "Fig 2.6  (rain gauge)"),
]

SEP = "─" * 70


def print_result(label: str, images: list[dict]) -> None:
    print(f"\n  → {len(images)} image(s) returned:")
    for i, img in enumerate(images, 1):
        url = img.get("url", "")
        cap = (img.get("caption") or "")[:80]
        rel = img.get("relevance", 0)
        clip = img.get("clip_similarity")
        clip_str = f"  clip={clip:.3f}" if clip is not None else ""
        print(f"    [{i}] relevance={rel:.1f}{clip_str}  {url}")
        if cap:
            print(f"         caption: {cap}")
    if not images:
        print("    (none — retrieval returned empty)")


def score_breakdown(query: str, upload_id: str) -> None:
    """Print per-image score breakdown using the debug API."""
    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookImage, TextbookUpload
    from sqlalchemy import select
    from app.services.image_service.image_intent_extractor import extract_image_intent
    from app.services.image_service.textbook_image_retrieval import (
        _hard_concept_filter,
        _concept_specificity_score,
        filter_chapter_candidates,
        _pages_by_upload_from_docs,
    )
    from app.services.image_service.figure_context_gates import (
        is_minimal_figure_caption,
        figure_descriptive_text_for_gates,
        context_supports_topic,
    )
    from app.services.image_service.symbolic_image_filters import apply_symbolic_hard_filters
    from app.config import SIDEBAR_MIN_SPECIFICITY, MIN_TOPIC_PURITY

    uid = uuid.UUID(upload_id)
    with SessionLocal() as db:
        images = db.scalars(
            select(TextbookImage)
            .where(TextbookImage.textbook_upload_id == uid)
            .order_by(TextbookImage.page_index, TextbookImage.sequence)
        ).all()

    intent = extract_image_intent(query, [])

    print(f"\n  Intent: core_concept={intent.core_concept!r}")
    print(f"          concept_tokens={sorted(intent.concept_tokens)}")
    print(f"          preferred_types={intent.preferred_types}")
    print(f"          required_terms={intent.required_terms[:5]}")

    print(f"\n  Per-image gate analysis ({len(images)} images):")
    for im in images:
        spec = _concept_specificity_score(intent, im)
        is_min = is_minimal_figure_caption(im.caption)
        gate_text = figure_descriptive_text_for_gates(im, im.caption_normalized or "")
        ctx_ok = context_supports_topic(intent, gate_text)
        reject, reason = _hard_concept_filter(intent, im)

        role = im.educational_role or "unknown"
        sidebar_blocked = (
            role in ("sidebar_example", "decorative")
            and spec < SIDEBAR_MIN_SPECIFICITY
            and not (is_min and ctx_ok)
        )

        status = "REJECT" if reject else "PASS  "
        fig = f"Fig {im.figure_number}" if im.figure_number else "no-fig"
        print(
            f"    {status} p{im.page_index:02d} {fig:8} spec={spec:5.1f} "
            f"role={role:20} min_cap={is_min} ctx_ok={ctx_ok} "
            f"sidebar_blocked={sidebar_blocked}"
            + (f"  ← {reason}" if reject else "")
        )


def main() -> None:
    print(SEP)
    print("WEATHER CHAPTER RETRIEVAL TEST")
    print(f"Upload: {UPLOAD_ID}")
    print(SEP)

    # First show score breakdown for both queries
    for query, expected in QUERIES:
        print(f"\nQUERY: {query!r}")
        print(f"Expected: {expected}")
        score_breakdown(query, UPLOAD_ID)

    print(f"\n{SEP}")
    print("LIVE RETRIEVAL RESULTS")
    print(SEP)

    from app.services.image_service.textbook_image_retrieval import related_images_for_query

    for query, expected in QUERIES:
        print(f"\nQ: {query!r}")
        print(f"   Expected → {expected}")
        try:
            results = related_images_for_query(
                text_collection_name=COLLECTION,
                chapter_ids=CHAPTER_IDS,
                chapter_names=["Chapter 2 - Understanding the Weather"],
                chapter_single="Chapter 2 - Understanding the Weather",
                query=query,
                retrieved_docs=[],
                top_n=3,
            )
            print_result(expected, results)
        except Exception as exc:
            print(f"   ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
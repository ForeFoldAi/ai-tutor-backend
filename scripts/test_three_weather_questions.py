"""
Test three weather-chapter questions: scope, images, and LLM answer shape.

Run from ai-tutor-backend:
  .venv/bin/python scripts/test_three_weather_questions.py
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

QUESTIONS = [
    "what is weather",
    "what are the Weather Instruments",
    "what is Weather Stations",
]

SEP = "=" * 72


def _answer_shape(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return "empty"
    has_bold_blocks = a.count("**") >= 4
    has_bullets = "•" in a or "\n-" in a or "\n* " in a
    emoji_sections = sum(1 for e in ("🌱", "📚", "🌍", "📝", "❓") if e in a)
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    plain_heads = sum(
        1
        for ln in lines
        if ln
        in (
            "Temperature",
            "Precipitation",
            "Atmospheric pressure",
            "Wind",
            "Humidity",
        )
        or ln.startswith("**")
    )
    return (
        f"chars={len(a)} bold_blocks={has_bold_blocks} bullets={has_bullets} "
        f"emoji_sections={emoji_sections} heading_lines≈{plain_heads}"
    )


async def main() -> None:
    from app.services.chat_service import _fetch_related_images, chapter_aware_qa
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.section_heading import subtopics_detail_for_main_section
    from app.services.section_retrieval import retrieve_for_tutor_query

    print(SEP)
    print("THREE-QUESTION WEATHER CHAPTER TEST")
    print(f"upload={UPLOAD_ID}")
    print(SEP)

    for q in QUESTIONS:
        print(f"\n{SEP}\nQ: {q!r}\n{SEP}")

        conv = resolve_conversation_context(q, chapter=CHAPTER)
        print(f"  context: mode={conv.response_mode.value} visual={conv.visual_intent.value}")

        docs, scope, instr = retrieve_for_tutor_query(
            q,
            collection_name=COLLECTION,
            chapter_ids=[UPLOAD_ID],
        )
        matched = scope.matched.title if scope.matched else None
        print(f"  scope: {scope.kind} matched={matched!r} pages={scope.page_start}-{scope.page_end}")

        subs = subtopics_detail_for_main_section(scope, docs)
        if subs:
            print(f"  subtopics ({len(subs)}): {[s.title for s in subs]}")

        img_ok = should_retrieve_images(conv, chapter_ids=[UPLOAD_ID], heading_scope_kind=scope.kind)
        print(f"  images_allowed: {img_ok}")

        imgs: list[dict] = []
        if img_ok:
            try:
                imgs = await asyncio.to_thread(
                    _fetch_related_images,
                    scope,
                    collection_name=COLLECTION,
                    chapter_ids=[UPLOAD_ID],
                    chapter_names=[CHAPTER],
                    chapter=CHAPTER,
                    query=q,
                    docs=docs,
                    conversation_history=None,
                    top_n=8,
                )
            except Exception as exc:
                print(f"  images ERROR: {exc}")
        print(f"  images ({len(imgs)}):")
        for i, im in enumerate(imgs, 1):
            cap = (im.get("caption") or "")[:70]
            print(
                f"    [{i}] subtopic={im.get('subtopic')!r} fig={im.get('figure_number')} "
                f"page={im.get('page')} rel={im.get('relevance')} | {cap}"
            )

        print("\n  --- LLM answer (may take ~15-30s) ---")
        try:
            answer, qa_imgs = await chapter_aware_qa(
                q,
                collection_name=COLLECTION,
                chapter_ids=[UPLOAD_ID],
                class_level="CLASS_9",
                board="CBSE",
                subject_name="Social",
                chapter=CHAPTER,
                chapter_names=[CHAPTER],
            )
            print(f"  shape: {_answer_shape(answer)}")
            print(f"  qa_images: {len(qa_imgs)}")
            preview = answer[:500].replace("\n", "\\n")
            if len(answer) > 500:
                preview += "..."
            print(f"  preview: {preview}")
        except Exception as exc:
            print(f"  LLM ERROR: {exc}")

    print(f"\n{SEP}\nDONE\n{SEP}")


if __name__ == "__main__":
    asyncio.run(main())

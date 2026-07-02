"""
Test Class 9 Science Chapter 1 — LLM answers + related image retrieval.

Run:
  cd ai-tutor-backend && .venv/bin/python scripts/test_science_chapter_qa.py
  cd ai-tutor-backend && .venv/bin/python scripts/test_science_chapter_qa.py --images-only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

UPLOAD_ID = "797fe8bf-c300-4a46-9fe6-f91b8ed4e6ba"
COLLECTION = "CBSE_CLASS_9_Science"
CHAPTER = "Chapter 1 - The Ever-Evolving World of Science"
CLASS_LEVEL = "CLASS_9"

# Questions mapped to chapter themes (Grade 7 Curiosity Science Ch.1 content in DB)
QUESTIONS = [
    "what is science",
    "what is the scientific method",
    "why do we ask questions in science",
    "what is curiosity in science",
    "what is observation in science",
    "what is an experiment",
    "what is a hypothesis",
    "how do scientists explore the world",
    "what is exploration in science",
    "why is science a process",
    "what are patterns in nature",
    "how do things work in science",
    "what is the ever-evolving world of science",
    "what skills do scientists need",
    "what is asking amazing questions in science",
    "show me a diagram about the scientific process",
    "explain laboratory and field work in science",
    "what is the role of experiments in science",
]

SEP = "=" * 72


async def main(*, images_only: bool = False, run_llm: bool = True) -> None:
    from app.services.chat_service import _fetch_related_images, chapter_aware_qa
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.section_retrieval import retrieve_for_tutor_query

    print(SEP)
    print("SCIENCE CHAPTER 1 — LLM + IMAGE TEST")
    print(f"upload={UPLOAD_ID}")
    print(SEP)

    results: list[tuple[str, int, bool, str]] = []

    for q in QUESTIONS:
        print(f"\n{SEP}\nQ: {q!r}\n{SEP}")

        conv = resolve_conversation_context(q, chapter=CHAPTER)
        docs, scope, _instr = retrieve_for_tutor_query(
            q,
            collection_name=COLLECTION,
            chapter_ids=[UPLOAD_ID],
            k=8,
        )
        print(f"  scope: {scope.kind} pages={scope.page_start}-{scope.page_end}")

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
                    top_n=3,
                )
            except Exception as exc:
                print(f"  images ERROR: {exc}")

        print(f"  images ({len(imgs)}):")
        for i, im in enumerate(imgs, 1):
            cap = (im.get("caption") or "")[:70]
            print(
                f"    [{i}] page={im.get('page')} file={im.get('url','').split('/')[-1]} "
                f"rel={im.get('relevance')} | {cap}"
            )

        llm_note = "skipped"
        if not images_only and run_llm:
            print("\n  --- LLM answer ---")
            try:
                answer, qa_imgs, _lesson = await chapter_aware_qa(
                    q,
                    collection_name=COLLECTION,
                    chapter_ids=[UPLOAD_ID],
                    class_level=CLASS_LEVEL,
                    board="CBSE",
                    subject_name="Science",
                    chapter=CHAPTER,
                    chapter_names=[CHAPTER],
                )
                print(f"  answer_chars={len(answer)} qa_images={len(qa_imgs)}")
                llm_note = "ok"
                preview = answer[:280].replace("\n", " ")
                print(f"  preview: {preview}...")
            except Exception as exc:
                llm_note = str(exc)[:80]
                print(f"  LLM ERROR: {exc}")

        case_ok = len(imgs) > 0
        results.append((q, len(imgs), case_ok, llm_note))

    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    with_imgs = sum(1 for _, n, _, _ in results if n > 0)
    print(f"  Questions with ≥1 image: {with_imgs}/{len(results)}")
    for q, n, ok, note in results:
        print(f"  {'IMG' if n else 'NO '} ({n}) {q!r}  llm={note}")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(images_only=args.images_only, run_llm=not args.no_llm))

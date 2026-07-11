"""
Multi-subject textbook QA test — 20 questions each for Social, Science, English.

Run:
  cd ai-tutor-backend && .venv/bin/python scripts/test_textbook_qa_multi_subject.py
  cd ai-tutor-backend && .venv/bin/python scripts/test_textbook_qa_multi_subject.py --subject social
  cd ai-tutor-backend && .venv/bin/python scripts/test_textbook_qa_multi_subject.py --images-only --no-llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

SEP = "=" * 72


@dataclass(frozen=True)
class TextbookConfig:
    key: str
    upload_id: str
    collection: str
    chapter: str
    class_level: str
    subject_name: str
    questions: tuple[str, ...]


TEXTBOOKS: tuple[TextbookConfig, ...] = (
    TextbookConfig(
        key="science",
        upload_id="797fe8bf-c300-4a46-9fe6-f91b8ed4e6ba",
        collection="CBSE_CLASS_9_Science",
        chapter="Chapter 1 - The Ever-Evolving World of Science",
        class_level="CLASS_9",
        subject_name="Science",
        questions=(
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
            "what is the ever-evolving world of science",
            "what is light and shadow",
            "what is eclipses",
            "what is day and night",
            "show me a diagram of a sunflower with roots",
            "what does a sunflower plant look like",
            "what are reversible and irreversible changes",
            "what properties of materials are discussed",
            "what skills do scientists need",
        ),
    ),
    TextbookConfig(
        key="social",
        upload_id="4851c848-a15b-4a45-b26c-995a52406797",
        collection="CBSE_CLASS_9_Social",
        chapter="Chapter 3 - Climates of India",
        class_level="CLASS_9",
        subject_name="Social",
        questions=(
            "what is climate",
            "what are seasons or ritu",
            "what are tropical and subtropical regions",
            "what are the tropics",
            "how does solar radiation affect climate",
            "show me diagram of solar radiation on earth",
            "what is monsoon climate in India",
            "what are winter festivals in India",
            "what is cyclone fani",
            "what is the eye of the storm",
            "what are cyclones and landslides",
            "what are Aravallis and urban heat islands",
            "how do seasons affect crops and clothing",
            "what is the climate of India",
            "what causes different climates in India",
            "show me a map of India climate",
            "what is humidity in climate",
            "what is temperature variation in India",
            "how does atmosphere affect climate",
            "what natural disasters are linked to climate",
        ),
    ),
    TextbookConfig(
        key="english",
        upload_id="c48355ac-9968-4834-b965-f37b076cf49b",
        collection="CBSE_CLASS_9_English",
        chapter="CHAPTER - 1",
        class_level="CLASS_9",
        subject_name="English",
        questions=(
            "what is the story How I Taught My Grandmother to Read about",
            "who is the author of How I Taught My Grandmother to Read",
            "why could the grandmother not read",
            "what is Kashi in the story",
            "how did the grandmother learn to read",
            "what kannada story did grandmother love",
            "what is adult literacy",
            "why is student participation important in literacy camps",
            "what states are mentioned in the national anthem",
            "what rivers are mentioned in the national anthem",
            "what mountain ranges are in the national anthem",
            "what is the theme of the grandmother story",
            "what reading habits are shown in the chapter",
            "what is the difference between past tense and past perfect",
            "what writing task is given about adult literacy",
            "who is the protagonist in the grandmother story",
            "what lesson does the grandmother story teach",
            "what serial story did grandmother follow",
            "what role does curiosity play in the english chapter",
            "what comprehension questions are in chapter 1",
        ),
    ),
)


async def run_textbook(
    cfg: TextbookConfig,
    *,
    images_only: bool,
    run_llm: bool,
) -> list[dict]:
    from app.services.chat_service import _fetch_related_images, chapter_aware_qa
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.section_retrieval import retrieve_for_tutor_query

    print(f"\n{SEP}\n{cfg.subject_name.upper()} — {cfg.chapter}\nupload={cfg.upload_id}\n{SEP}")

    results: list[dict] = []

    for q in cfg.questions:
        print(f"\nQ: {q!r}")

        conv = resolve_conversation_context(q, chapter=cfg.chapter)
        docs, scope, _instr = retrieve_for_tutor_query(
            q,
            collection_name=cfg.collection,
            chapter_ids=[cfg.upload_id],
            k=8,
        )
        img_ok = should_retrieve_images(
            conv, chapter_ids=[cfg.upload_id], heading_scope_kind=scope.kind
        )

        imgs: list[dict] = []
        if img_ok:
            try:
                imgs = await asyncio.to_thread(
                    _fetch_related_images,
                    scope,
                    collection_name=cfg.collection,
                    chapter_ids=[cfg.upload_id],
                    chapter_names=[cfg.chapter],
                    chapter=cfg.chapter,
                    query=q,
                    docs=docs,
                    conversation_history=None,
                    top_n=3,
                )
            except Exception as exc:
                print(f"  images ERROR: {exc}")

        img_files = [
            {
                "file": (im.get("url") or "").split("/")[-1],
                "page": im.get("page"),
                "relevance": im.get("relevance"),
                "caption": (im.get("caption") or "")[:80],
            }
            for im in imgs
        ]
        print(f"  images={len(imgs)} scope={scope.kind}")
        for i, im in enumerate(img_files, 1):
            print(f"    [{i}] p{im['page']} {im['file']} rel={im['relevance']} | {im['caption']}")

        llm_note = "skipped"
        answer_preview = ""
        qa_img_count = 0
        if not images_only and run_llm:
            try:
                answer, qa_imgs, _lesson = await chapter_aware_qa(
                    q,
                    collection_name=cfg.collection,
                    chapter_ids=[cfg.upload_id],
                    class_level=cfg.class_level,
                    board="CBSE",
                    subject_name=cfg.subject_name,
                    chapter=cfg.chapter,
                    chapter_names=[cfg.chapter],
                )
                llm_note = "ok"
                qa_img_count = len(qa_imgs)
                answer_preview = answer[:200].replace("\n", " ")
                print(f"  LLM ok chars={len(answer)} qa_images={qa_img_count}")
            except Exception as exc:
                llm_note = str(exc)[:120]
                print(f"  LLM ERROR: {exc}")

        results.append(
            {
                "question": q,
                "image_count": len(imgs),
                "images": img_files,
                "llm": llm_note,
                "qa_image_count": qa_img_count,
                "answer_preview": answer_preview,
            }
        )

    with_imgs = sum(1 for r in results if r["image_count"] > 0)
    llm_ok = sum(1 for r in results if r["llm"] == "ok")
    print(f"\n{SEP}\n{cfg.subject_name} SUMMARY: images {with_imgs}/{len(results)} | LLM ok {llm_ok}/{len(results)}\n{SEP}")
    for r in results:
        flag = "IMG" if r["image_count"] else "NO "
        print(f"  {flag} ({r['image_count']}) {r['question']!r} llm={r['llm']}")

    return results


async def main(*, subject: str | None, images_only: bool, run_llm: bool, out_json: str | None) -> None:
    selected = [t for t in TEXTBOOKS if subject is None or t.key == subject]
    if not selected:
        raise SystemExit(f"Unknown subject {subject!r}; use science, social, or english")

    all_results: dict[str, list[dict]] = {}
    for cfg in selected:
        all_results[cfg.key] = await run_textbook(cfg, images_only=images_only, run_llm=run_llm)

    print(f"\n{SEP}\nOVERALL\n{SEP}")
    for key, rows in all_results.items():
        with_imgs = sum(1 for r in rows if r["image_count"] > 0)
        llm_ok = sum(1 for r in rows if r["llm"] == "ok")
        print(f"  {key}: images {with_imgs}/20 | LLM {llm_ok}/20")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=["science", "social", "english"])
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    asyncio.run(
        main(
            subject=args.subject,
            images_only=args.images_only,
            run_llm=not args.no_llm,
            out_json=args.out_json or None,
        )
    )

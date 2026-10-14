"""Smoke: text + voice answers/images for map / political map / Mughals.

Run:
  .venv/bin/python scripts/smoke_map_mughal_images.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"), override=False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("app.services.image_service.llm_image_select").setLevel(logging.INFO)

UPLOAD_ID = "33"
COLLECTION = "CBSE_CLASS_8_Social"
CHAPTER = "Chapter 2 - Reshaping India's Political Map"
QUESTIONS = [
    "what is map?",
    "what is political map?",
    "Who are mughuls?",
]


def _img_summary(imgs: list[dict] | None) -> list[dict]:
    out = []
    for im in imgs or []:
        out.append(
            {
                "figure": im.get("figure_number"),
                "page": im.get("page"),
                "caption": (im.get("caption") or "")[:90],
                "file": im.get("file_name"),
            }
        )
    return out


async def run_text(q: str, kw: dict):
    from app.services.chat_service import chapter_aware_qa

    answer, images, *_rest = await chapter_aware_qa(q, **kw)
    return answer, images


async def run_voice(q: str, kw: dict):
    from app.services.chat_service import chapter_aware_qa_stream

    parts: list[str] = []
    batches: list[list[dict]] = []

    async def capture(imgs: list[dict]) -> None:
        batches.append(list(imgs or []))

    async for tok in chapter_aware_qa_stream(
        q, voice_mode=True, emit_related_images=capture, **kw
    ):
        parts.append(tok)
    final = batches[-1] if batches else []
    return "".join(parts), final, len(batches)


async def main() -> None:
    from app.config import ENABLE_LLM_IMAGE_SELECT

    print("ENABLE_LLM_IMAGE_SELECT=", ENABLE_LLM_IMAGE_SELECT)
    print("chapter=", CHAPTER, "upload_id=", UPLOAD_ID)

    kw = dict(
        collection_name=COLLECTION,
        chapter_ids=[UPLOAD_ID],
        class_level="CLASS_8",
        board="CBSE",
        subject_name="Social",
        chapter=CHAPTER,
        chapter_names=[CHAPTER],
    )

    for q in QUESTIONS:
        print("\n" + "=" * 72)
        print("Q:", q)
        print("--- TEXT-TO-TEXT ---")
        try:
            ans, imgs = await run_text(q, kw)
            print("answer:", (ans or "")[:450].replace("\n", " "))
            print(f"images({len(imgs or [])}):", _img_summary(imgs))
        except Exception as exc:
            print("TEXT FAIL", type(exc).__name__, exc)

        print("--- AI VOICE (voice_mode stream) ---")
        try:
            ans, imgs, n_emit = await run_voice(q, kw)
            print("answer:", (ans or "")[:450].replace("\n", " "))
            print(f"emits={n_emit} final_images({len(imgs or [])}):", _img_summary(imgs))
        except Exception as exc:
            print("VOICE FAIL", type(exc).__name__, exc)


if __name__ == "__main__":
    asyncio.run(main())

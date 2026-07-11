#!/usr/bin/env python3
"""Backfill vision captions for textbook figures with weak PDF/OCR metadata.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/backfill_vision_captions.py --upload-id <uuid>
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/backfill_vision_captions.py --upload-id <uuid> --reindex
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

load_dotenv(BACKEND_ROOT / ".env")

from sqlalchemy import select

from app.core.database import SessionLocal
from app.modules.catalog.models import TextbookImage, TextbookUpload
from app.services.image_service.textbook_image_extraction import image_disk_path
from app.services.image_service.vision_caption_service import (
    apply_vision_caption_to_row,
    is_weak_text_caption,
    vision_caption_available,
)


def backfill_upload(upload_id: str, *, reindex: bool = False, force: bool = False) -> int:
    if not vision_caption_available():
        print("Vision caption model unavailable. Check VISION_CAPTION_ENABLED and HF_TOKEN.")
        return 1

    db = SessionLocal()
    try:
        upload = db.get(TextbookUpload, upload_id)
        if upload is None:
            print(f"Upload not found: {upload_id}")
            return 1

        rows = list(
            db.scalars(
                select(TextbookImage)
                .where(TextbookImage.textbook_upload_id == upload.id)
                .order_by(TextbookImage.page_index, TextbookImage.sequence)
            ).all()
        )
        updated = 0
        for row in rows:
            if (row.content_kind or "figure") != "figure":
                continue
            effective = (row.caption or row.generated_caption or "").strip()
            if not force and not is_weak_text_caption(effective):
                continue
            path = image_disk_path(upload.id, row.file_name)
            if not os.path.isfile(path):
                print(f"  skip missing file: {row.file_name}")
                continue
            with open(path, "rb") as fh:
                image_bytes = fh.read()
            if apply_vision_caption_to_row(
                row,
                image_bytes=image_bytes,
                upload=upload,
                nearby_before=row.nearby_text_before_figure or "",
                nearby_after=row.nearby_text_after_figure or "",
            ):
                updated += 1
                cap = (row.generated_caption or "")[:90]
                print(f"  p{row.page_index:02d} {row.file_name}: {cap}")

        db.commit()
        print(f"Updated {updated}/{len(rows)} images for upload {upload_id}")

        if reindex and updated > 0:
            from app.services.image_service.figure_context_bge import index_figure_context_embeddings
            from app.services.image_service.multimodal_image_index import index_upload_images

            for row in rows:
                row.context_bge_indexed = False
                row.multimodal_indexed = False
            db.commit()

            n_bge = index_figure_context_embeddings(db, upload)
            n_clip = index_upload_images(db, upload)
            db.commit()
            print(f"Reindexed: BGE={n_bge} CLIP={n_clip}")

        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill BLIP vision captions for textbook figures")
    parser.add_argument("--upload-id", type=str, required=True)
    parser.add_argument("--reindex", action="store_true", help="Re-index BGE + CLIP after backfill")
    parser.add_argument("--force", action="store_true", help="Re-caption even when text caption looks OK")
    args = parser.parse_args()
    return backfill_upload(args.upload_id, reindex=args.reindex, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill CLIP embeddings for all textbook images on disk.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/backfill_multimodal_images.py
"""

from __future__ import annotations

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
from app.modules.catalog.models import TextbookUpload
from app.services.image_service.multimodal_image_index import index_upload_images
from app.services.image_service.multimodal_encoder import clip_model_available


def main() -> int:
    if not clip_model_available():
        print("FAIL: CLIP model could not be loaded. Check MULTIMODAL_IMAGE_MODEL and dependencies.")
        return 1

    db = SessionLocal()
    try:
        uploads = list(
            db.scalars(
                select(TextbookUpload)
                .where(TextbookUpload.file_path.isnot(None))
                .order_by(TextbookUpload.upload_date.desc())
            )
        )
        print(f"Found {len(uploads)} uploads with file_path")
        total = 0
        for u in uploads:
            n = index_upload_images(db, u)
            total += n
            print(f"  {u.id} → indexed {n} images")
        print(f"Done. Total images indexed: {total}")
        return 0 if total >= 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

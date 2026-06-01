#!/usr/bin/env python3
"""Re-index BGE (figure_context) and CLIP for textbook images after layout migration.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/reindex_textbook_figure_context.py
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/reindex_textbook_figure_context.py --upload-id <uuid>
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
from app.services.figure_context_bge import index_figure_context_embeddings
from app.services.multimodal_image_index import index_upload_images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-id", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.upload_id:
            uploads = [db.get(TextbookUpload, args.upload_id)]
            if uploads[0] is None:
                print(f"Not found: {args.upload_id}")
                return 1
        elif args.all:
            upload_ids = db.scalars(
                select(TextbookImage.textbook_upload_id).distinct()
            ).all()
            uploads = [db.get(TextbookUpload, uid) for uid in upload_ids]
            uploads = [u for u in uploads if u is not None]
        else:
            parser.print_help()
            return 1

        for upload in uploads:
            rows = list(
                db.scalars(
                    select(TextbookImage).where(TextbookImage.textbook_upload_id == upload.id)
                ).all()
            )
            for row in rows:
                row.context_bge_indexed = False
            db.commit()

            clip_n = index_upload_images(db, upload)
            bge_n = index_figure_context_embeddings(db, upload)
            print(f"{upload.id}: CLIP={clip_n} BGE={bge_n}")

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

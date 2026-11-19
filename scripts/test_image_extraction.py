#!/usr/bin/env python3
"""
Run from repo backend root:
  cd ai-tutor-backend && .venv/bin/python scripts/test_image_extraction.py [partial_filename]

Loads .env, finds a TextbookUpload (by optional substring), purges + re-extracts images,
prints counts and sample paths. Exit 0 if at least one image was created.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)

from dotenv import load_dotenv

load_dotenv(BACKEND_ROOT / ".env")

from sqlalchemy import select, func

from app.core.database import SessionLocal
from app.modules.catalog.models import TextbookImage, TextbookUpload
from app.services.image_service.textbook_image_extraction import (
    IMAGE_ROOT,
    image_count_for_upload,
    purge_textbook_images_disk_and_rows,
    ensure_textbook_images_extracted,
)


def main() -> int:
    needle = (sys.argv[1] if len(sys.argv) > 1 else ".pdf").strip()
    db = SessionLocal()
    try:
        row = db.scalar(
            select(TextbookUpload)
            .where(TextbookUpload.file_path.isnot(None))
            .where(TextbookUpload.file_path.ilike(f"%{needle}%"))
            .order_by(TextbookUpload.upload_date.desc())
            .limit(1)
        )
        if not row:
            print(f"FAIL: no textbook_uploads row with file_path ilike '%{needle}%'")
            return 2

        print("Upload:", row.id)
        print("file_path:", row.file_path)
        fp = Path(row.file_path or "")
        print("file exists:", fp.is_file(), "size:", fp.stat().st_size if fp.is_file() else 0)

        before = image_count_for_upload(db, row.id)
        print("textbook_images rows before:", before)

        print("Purging old images + disk…")
        purge_textbook_images_disk_and_rows(db, row.id)
        db.commit()

        print("Running extraction…")
        n = ensure_textbook_images_extracted(db, row)
        print("extract_pdf_images / extract_docx_images returned:", n)

        after = image_count_for_upload(db, row.id)
        print("textbook_images rows after:", after)

        img_dir = Path(IMAGE_ROOT) / str(row.id)
        if img_dir.is_dir():
            files = sorted(img_dir.rglob("*.jpg"))
            print("files on disk:", len(files))
            for p in files[:8]:
                rel = p.relative_to(img_dir)
                print("  ", rel, p.stat().st_size, "bytes")
            if len(files) > 8:
                print("  …")
        else:
            print("IMAGE_DIR (missing):", img_dir)

        if after > 0:
            samples = db.scalars(
                select(TextbookImage)
                .where(TextbookImage.textbook_upload_id == row.id)
                .order_by(TextbookImage.page_index, TextbookImage.sequence)
                .limit(5)
            )
            for im in samples:
                print("  caption:", (im.caption or "(none)")[:100])
            print("RESULT: OK — extraction produced", after, "figure(s) with captions.")
            return 0
        print("RESULT: FAIL — zero images (check logs above / PDF has no extractable images).")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

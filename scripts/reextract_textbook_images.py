#!/usr/bin/env python3
"""Force layout-aware re-extraction for textbook uploads.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/reextract_textbook_images.py
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/reextract_textbook_images.py --upload-id <uuid>
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/reextract_textbook_images.py --all-pdf
"""

from __future__ import annotations

import argparse
import json
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
from app.services.image_service.pdf_extraction_types import extract_document_layout, validation_report_dict
from app.services.image_service.textbook_image_extraction import reextract_textbook_images


def _write_validation_report(upload_id: str, pdf_path: str, out_dir: Path) -> Path:
    result = extract_document_layout(pdf_path, max_figures=96)
    report = validation_report_dict(result)
    report["upload_id"] = upload_id
    report["pdf_path"] = pdf_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"validation_{upload_id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-extract textbook images (layout-aware)")
    parser.add_argument("--upload-id", type=str, help="Single upload UUID")
    parser.add_argument("--all-pdf", action="store_true", help="Re-extract all PDF uploads")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(BACKEND_ROOT / "uploads" / "_extraction_reports"),
        help="Directory for extraction JSON reports",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validation report only, no DB changes")
    parser.add_argument("--pdf-path", type=str, help="PDF path for validation-only (no DB)")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)

    if args.pdf_path:
        pdf = Path(args.pdf_path)
        if not pdf.is_file():
            print(f"File not found: {pdf}")
            return 1
        uid = args.upload_id or pdf.stem
        report_path = _write_validation_report(uid, str(pdf), report_dir)
        print(f"Validation report: {report_path}")
        return 0

    db = SessionLocal()
    try:
        if args.upload_id:
            uploads = [db.get(TextbookUpload, args.upload_id)]
            if uploads[0] is None:
                print(f"Upload not found: {args.upload_id}")
                return 1
        elif args.all_pdf:
            uploads = list(
                db.scalars(
                    select(TextbookUpload)
                    .where(TextbookUpload.file_path.isnot(None))
                    .order_by(TextbookUpload.upload_date.desc())
                )
            )
            uploads = [u for u in uploads if (u.file_path or "").lower().endswith(".pdf")]
        else:
            parser.print_help()
            return 1

        total = 0
        for upload in uploads:
            path = upload.file_path or ""
            if not path or not os.path.isfile(path):
                print(f"SKIP {upload.id}: missing file")
                continue

            report_path = _write_validation_report(str(upload.id), path, report_dir)
            print(f"Validation report: {report_path}")

            if args.dry_run:
                continue

            n = reextract_textbook_images(db, upload)
            total += n
            print(f"Re-extracted {upload.id}: {n} figures")

        print(f"Done. Total figures: {total}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

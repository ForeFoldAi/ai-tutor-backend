#!/usr/bin/env python3
"""Write student captions for existing figures with the vision LLM (Phase 4 backfill).

Any board / class / subject. Resumable: figures already captioned by the VLM are skipped
unless --force. Non-teaching unnumbered figures (mascots, icons, text boxes) are hidden;
numbered figures the model sees as text are tagged ``needs_recrop``.

Uses the app's "vision" model settings (LLM_VISION_MODEL / LLM_VISION_BASE_URL /
LLM_VISION_API_KEY). Provider rate limits (429) are waited out automatically.

  cd ai-tutor-backend && .venv/bin/python scripts/caption_figures.py --board CBSE --class-level CLASS_10 --limit 5 --dry-run
  cd ai-tutor-backend && .venv/bin/python scripts/caption_figures.py --board CBSE --class-level CLASS_10
"""

from __future__ import annotations

import argparse
import json
import logging
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
from app.services.image_service.figure_caption_vlm import caption_upload_figures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board")
    ap.add_argument("--class-level")
    ap.add_argument("--subject")
    ap.add_argument("--upload-id", type=int, action="append")
    ap.add_argument("--limit", type=int, help="max figures per chapter")
    ap.add_argument("--force", action="store_true", help="re-caption figures already captioned")
    ap.add_argument("--dry-run", action="store_true", help="print captions, write nothing")
    ap.add_argument("--out", help="also write every result to this JSONL file")
    args = ap.parse_args()
    if not (args.board or args.class_level or args.subject or args.upload_id):
        ap.error("pass a filter (--board/--class-level/--subject/--upload-id)")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app.services.image_service.figure_caption_vlm").setLevel(logging.INFO)

    with SessionLocal() as db:
        q = select(TextbookUpload.id, TextbookUpload.subject_name, TextbookUpload.chapter).order_by(TextbookUpload.id)
        if args.upload_id:
            q = q.where(TextbookUpload.id.in_(args.upload_id))
        if args.board:
            q = q.where(TextbookUpload.board == args.board)
        if args.class_level:
            q = q.where(TextbookUpload.class_level == args.class_level)
        if args.subject:
            q = q.where(TextbookUpload.subject_name == args.subject)
        uploads = list(db.execute(q))

    out = open(args.out, "a", encoding="utf-8") if args.out else None

    def on_result(row, result) -> None:
        line = {"image_id": row.id, "upload_id": row.textbook_upload_id, "figure_number": row.figure_number,
                "result": result.__dict__ if result else None}
        if out:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()
        if result:
            flag = "" if result.is_teaching_figure else " [not teaching]"
            print(f"    #{row.id} fig {row.figure_number or '—'}{flag}: {result.caption}", flush=True)
        else:
            print(f"    #{row.id} fig {row.figure_number or '—'}: (no usable caption)", flush=True)

    totals: dict[str, int] = {}
    for uid, subject, chapter in uploads:
        print(f"upload {uid} · {subject} · {chapter}", flush=True)
        stats = caption_upload_figures(uid, force=args.force, limit=args.limit,
                                       on_result=on_result, dry_run=args.dry_run)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
    print(f"\n{'DRY RUN ' if args.dry_run else ''}totals: {totals}")
    if out:
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Flag junk figures already in the database (QR codes, QR chapter banners, decorations
repeated across a book's chapters) as ``is_decorative`` so students never see them.

Also fills the missing ``phash`` column (it was never populated before the
``compute_phash`` fix).

Dry run by default — prints what would change and writes a JSON report. Nothing is
deleted; flagged rows stay visible to admins, who can unhide any of them.

  cd ai-tutor-backend && .venv/bin/python scripts/flag_junk_figures.py --board CBSE --class-level CLASS_10
  cd ai-tutor-backend && .venv/bin/python scripts/flag_junk_figures.py --board CBSE --class-level CLASS_10 --apply
  cd ai-tutor-backend && .venv/bin/python scripts/flag_junk_figures.py --all --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

load_dotenv(BACKEND_ROOT / ".env")

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import SessionLocal
from app.modules.catalog.models import TextbookImage, TextbookUpload
from app.services.image_service.junk_figure_filter import (
    REASON_RECURRING,
    BookFigure,
    classify_qr_junk,
    find_recurring_decorations,
)
from app.services.image_service.storage_backend import get_storage_backend
from app.services.image_service.textbook_image_extraction import compute_phash, image_storage_key

logger = logging.getLogger("flag_junk_figures")


def _book_key(up: TextbookUpload) -> tuple[str, str, str]:
    return (str(up.board.value if hasattr(up.board, "value") else up.board),
            str(up.class_level.value if hasattr(up.class_level, "value") else up.class_level),
            up.subject_name)


def _scan(row: TextbookImage, backend) -> tuple[int, int | None, str | None, bool]:
    """(id, phash, qr_reason, fetched)."""
    data = backend.get_bytes(image_storage_key(row.upload, row.file_name))
    if not data:
        return row.id, row.phash, None, False
    ph = row.phash if row.phash is not None else compute_phash(data)
    return row.id, ph, classify_qr_junk(data, figure_number=row.figure_number), True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board")
    ap.add_argument("--class-level", help="e.g. CLASS_10")
    ap.add_argument("--subject", help="exact subject_name")
    ap.add_argument("--all", action="store_true", help="every board/class/subject")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", default="", help="JSON report path (default: uploads/_reports/...)")
    args = ap.parse_args()
    if not (args.all or args.board or args.class_level or args.subject):
        ap.error("pass --all or at least one of --board/--class-level/--subject")
    logging.basicConfig(level=logging.WARNING)

    backend = get_storage_backend()
    db = SessionLocal()
    try:
        q = (
            select(TextbookImage)
            .join(TextbookUpload, TextbookUpload.id == TextbookImage.textbook_upload_id)
            .where(TextbookImage.content_kind == "figure")
            .options(selectinload(TextbookImage.upload))
        )
        if args.board:
            q = q.where(TextbookUpload.board == args.board)
        if args.class_level:
            q = q.where(TextbookUpload.class_level == args.class_level)
        if args.subject:
            q = q.where(TextbookUpload.subject_name == args.subject)
        rows = list(db.scalars(q))
        by_id = {r.id: r for r in rows}
        print(f"Scanning {len(rows)} figures…", flush=True)

        with ThreadPoolExecutor(max(1, args.workers)) as ex:
            scans = list(ex.map(lambda r: _scan(r, backend), rows))

        missing = sum(1 for *_, ok in scans if not ok)
        reasons: dict[int, str] = {}
        phash_filled = 0
        books: dict[tuple[str, str, str], list[BookFigure]] = defaultdict(list)
        for img_id, ph, qr_reason, _ in scans:
            row = by_id[img_id]
            if ph is not None and row.phash is None:
                phash_filled += 1
                if args.apply:
                    row.phash = ph
            if qr_reason:
                reasons[img_id] = qr_reason
            books[_book_key(row.upload)].append(
                BookFigure(key=img_id, upload_id=row.textbook_upload_id, phash=ph,
                           figure_number=row.figure_number)
            )
        for figs in books.values():
            for key in find_recurring_decorations(figs):
                reasons.setdefault(int(key), REASON_RECURRING)

        newly: dict[int, str] = {i: r for i, r in reasons.items() if not by_id[i].is_decorative}
        if args.apply:
            for img_id, reason in newly.items():
                row = by_id[img_id]
                row.is_decorative = True
                row.educational_role = "decorative"
                tags = [t for t in (row.educational_tags or "").split("|") if t and not t.startswith("junk:")]
                row.educational_tags = "|".join([f"junk:{reason}", *tags])[:256]
            db.commit()

        per_book = Counter()
        for i, r in newly.items():
            b = _book_key(by_id[i].upload)
            per_book[(*b, r)] += 1
        print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {len(newly)} figures newly flagged "
              f"({len(reasons) - len(newly)} already hidden); phash filled: {phash_filled}; "
              f"images missing in storage: {missing}")
        print("By reason:", dict(Counter(newly.values())))
        for (board, cls, subj, reason), n in sorted(per_book.items()):
            print(f"  {board:12} {cls:9} {subj[:38]:38} {reason:22} {n}")

        report = args.report or str(BACKEND_ROOT / "uploads" / "_reports" / "flag_junk_figures.json")
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps({
            "applied": args.apply,
            "scanned": len(rows),
            "missing_in_storage": missing,
            "phash_filled": phash_filled,
            "flagged": [
                {"image_id": i, "upload_id": by_id[i].textbook_upload_id, "page": by_id[i].page_index + 1,
                 "file_name": by_id[i].file_name, "reason": r}
                for i, r in sorted(newly.items())
            ],
        }, indent=1))
        print(f"Report: {report}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

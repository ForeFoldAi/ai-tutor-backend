#!/usr/bin/env python3
"""Recompute image_type / educational_role for existing figures (caption → vision summary → nearby text).

Older rows were typed from the whole figure context, so a chapter titled "...Political Map"
tagged every photo and painting as a map.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/retag_image_types.py [upload_id ...] [--apply]

Only changes a type when the caption or vision summary gives positive evidence; rows with
no evidence keep their old type until a vision summary is stored for them.
Dry run by default (prints a summary of changes).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logging.disable(logging.CRITICAL)

from sqlalchemy import select  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402
from app.modules.catalog.models import TextbookImage  # noqa: E402
from app.services.image_service.textbook_image_extraction import (  # noqa: E402
    classify_educational_role,
    classify_image_type,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("upload_ids", nargs="*", type=int)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    changes: Counter[tuple[str, str]] = Counter()
    role_changes = 0
    total = 0
    with SessionLocal() as db:
        q = select(TextbookImage).where(TextbookImage.content_kind == "figure")
        if args.upload_ids:
            q = q.where(TextbookImage.textbook_upload_id.in_(args.upload_ids))
        for row in db.scalars(q).yield_per(500):
            total += 1
            new_type = classify_image_type(
                row.caption or "", row.page_text_snippet or "", row.visual_summary or ""
            )
            if new_type == "unknown" or new_type == row.image_type:
                continue
            new_role = row.educational_role
            if row.educational_role not in ("decorative", "activity"):
                new_role = classify_educational_role(new_type, row.caption or row.visual_summary or "")
            if new_type != row.image_type:
                changes[(row.image_type, new_type)] += 1
            if new_role != row.educational_role:
                role_changes += 1
            if args.apply:
                row.image_type = new_type
                row.educational_role = new_role
        if args.apply:
            db.commit()

    print(f"figures scanned: {total}; type changes: {sum(changes.values())}; role changes: {role_changes}")
    for (old, new), n in changes.most_common(15):
        print(f"  {old:>16} -> {new:<16} {n}")
    print("applied" if args.apply else "dry run (use --apply)")


if __name__ == "__main__":
    main()

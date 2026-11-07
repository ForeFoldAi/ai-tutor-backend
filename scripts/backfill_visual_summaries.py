#!/usr/bin/env python3
"""Store a vision-model description (``visual_summary``) for every extracted figure.

Chat image selection reads the stored text instead of calling the vision model per request.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/backfill_visual_summaries.py [upload_id ...] [--overwrite]

No ids = every upload that has figures without a summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("app.services.image_service.llm_image_select").setLevel(logging.INFO)

from sqlalchemy import select  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402
from app.modules.catalog.models import TextbookImage  # noqa: E402
from app.services.image_service.llm_image_select import backfill_visual_summaries  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("upload_ids", nargs="*", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = args.upload_ids
    if not ids:
        with SessionLocal() as db:
            ids = sorted(
                set(
                    db.scalars(
                        select(TextbookImage.textbook_upload_id).where(
                            TextbookImage.content_kind == "figure",
                            TextbookImage.visual_summary.is_(None),
                        )
                    )
                )
            )
    total = 0
    for uid in ids:
        n = backfill_visual_summaries(uid, overwrite=args.overwrite)
        total += n
        print(f"upload {uid}: described {n}")
    print(f"done: {total} figures described across {len(ids)} uploads")


if __name__ == "__main__":
    main()

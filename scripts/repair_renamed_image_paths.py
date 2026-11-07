#!/usr/bin/env python3
"""Copy textbook images back under the prefix the app expects after a subject/chapter rename.

Image keys are ``{board}/{class}/{subject}/{chapter}/...``; renames made before the rename
handler copied objects left them under the old subject folder, so they 404 in chat.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/repair_renamed_image_paths.py [--apply] [--delete-old]

Dry run by default. Finds the old folder by matching board/class/chapter plus one real file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logging.disable(logging.CRITICAL)

from sqlalchemy import select  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402
from app.modules.catalog.models import TextbookImage, TextbookUpload  # noqa: E402
from app.services.image_service.storage_backend import S3StorageBackend, get_storage_backend  # noqa: E402
from app.services.image_service.textbook_image_extraction import image_storage_key, upload_asset_prefix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="copy objects (default: dry run)")
    ap.add_argument("--delete-old", action="store_true", help="delete the old prefix after copying")
    args = ap.parse_args()

    backend = get_storage_backend()
    if not isinstance(backend, S3StorageBackend):
        print("STORAGE_BACKEND is not s3; nothing to repair.")
        return 0
    client, bucket = backend._client, backend.bucket

    def subfolders(prefix: str) -> list[str]:
        out: list[str] = []
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=backend._key(prefix) + "/", Delimiter="/"
        ):
            out += [p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in page.get("CommonPrefixes") or ()]
        return out

    db = SessionLocal()
    uploads = db.scalars(
        select(TextbookUpload).where(TextbookUpload.id.in_(select(TextbookImage.textbook_upload_id).distinct()))
    ).all()

    fixed = unresolved = 0
    subject_cache: dict[str, list[str]] = {}
    def matches(rel: str, content_hash: str | None) -> bool:
        try:
            head = client.head_object(Bucket=bucket, Key=backend._key(rel))
        except Exception:
            return False
        stored = (head.get("Metadata") or {}).get("content-hash")
        return not content_hash or not stored or stored == content_hash

    for u in uploads:
        sample, sample_hash = db.execute(
            select(TextbookImage.file_name, TextbookImage.content_hash)
            .where(TextbookImage.textbook_upload_id == u.id)
            .limit(1)
        ).one()
        if backend.exists(image_storage_key(u, sample)) or backend.exists(image_storage_key(u.id, sample, legacy=True)):
            continue
        board, grade, _subject, chapter = upload_asset_prefix(u).split("/", 3)
        grade_prefix = f"{board}/{grade}"
        subjects = subject_cache.setdefault(grade_prefix, subfolders(grade_prefix))
        old = next(
            (f"{grade_prefix}/{s}/{chapter}" for s in subjects if matches(f"{grade_prefix}/{s}/{chapter}/{sample}", sample_hash)),
            None,
        )
        new = upload_asset_prefix(u)
        if old is None:
            unresolved += 1
            print(f"UNRESOLVED upload={u.id} expected={new}")
            continue
        print(f"{'COPY' if args.apply else 'WOULD COPY'} upload={u.id} {old} -> {new}")
        if args.apply:
            n = backend.copy_prefix(old, new)
            if args.delete_old:
                backend.delete_prefix(old)
            print(f"  copied {n} objects")
        fixed += 1

    print(f"\n{'repaired' if args.apply else 'repairable'}={fixed} unresolved={unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

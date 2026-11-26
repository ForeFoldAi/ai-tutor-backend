"""Re-index the TEXT of chapters whose vector index lost chunks. Images are untouched.

Before the batching fix, the upload pipeline indexed chapters in 50-chunk batches
and every batch deleted the chapter's earlier vectors, so any chapter with more than
50 chunks kept only its last batch (end-of-chapter exercises / transcripts).

This script re-extracts each chapter's PDF exactly like the upload pipeline does,
clears that chapter's vectors once, and re-inserts every chunk. It does NOT run the
image re-extraction step, so admin-edited figures, crops and captions are kept.

Usage (from ai-tutor-backend):
    .venv/bin/python scripts/repair_partial_index.py                 # dry run, auto-detect damaged chapters
    .venv/bin/python scripts/repair_partial_index.py --ids 752 753    # dry run for specific chapters
    .venv/bin/python scripts/repair_partial_index.py --apply          # actually rewrite the vectors

Dry run (the default) extracts and counts chunks but writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _indexed_count(client, coll: str, uid: int) -> int:
    from app.services.vector_backend.qdrant_text import textbook_upload_filter

    return client.count(coll, count_filter=textbook_upload_filter([str(uid)]), exact=True).count


def _damaged_ids(db, client, existing: set[str], min_ratio: float) -> list[int]:
    from sqlalchemy import text

    from app.services.chapter_collection import collection_name

    rows = db.execute(
        text(
            "SELECT id, board::text, class_level::text, subject_name, chunk_count "
            "FROM textbook_uploads WHERE chunk_count > 0 ORDER BY id"
        )
    ).fetchall()
    out = []
    for uid, board, cl, subject, chunk_count in rows:
        coll = collection_name(board or "", cl or "", subject or "")
        if coll not in existing:
            continue
        if _indexed_count(client, coll, uid) < min_ratio * int(chunk_count or 0):
            out.append(int(uid))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ids", nargs="*", type=int, default=None, help="textbook_uploads ids")
    ap.add_argument("--apply", action="store_true", help="write to the index (default: dry run)")
    ap.add_argument("--min-ratio", type=float, default=0.8)
    args = ap.parse_args()

    from sqlalchemy import text

    from app.core.database import SessionLocal
    from app.services.catalog_pipeline.constants import EMBEDDING_BATCH_SIZE
    from app.services.chapter_collection import collection_name
    from app.services.document_service import process_document
    from app.services.image_service.storage_backend import materialize_textbook_file
    from app.services.vector_backend.qdrant_text import get_qdrant_client
    from app.services.vector_service import add_documents_to_store, delete_collection_docs

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    db = SessionLocal()
    try:
        ids = args.ids if args.ids else _damaged_ids(db, client, existing, args.min_ratio)
        if not ids:
            print("No damaged chapters found.")
            return 0
        print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(ids)} chapter(s): {ids}")
        failures = 0
        for uid in ids:
            row = db.execute(
                text(
                    "SELECT id, board::text, class_level::text, subject_name, chapter, "
                    "content_type, content_label, file_path FROM textbook_uploads WHERE id=:id"
                ),
                {"id": uid},
            ).mappings().first()
            if not row:
                print(f"  id={uid}: not found")
                failures += 1
                continue
            coll = collection_name(row["board"], row["class_level"], row["subject_name"])
            before = _indexed_count(client, coll, uid) if coll in existing else 0
            local = Path(__file__).resolve().parents[1] / "uploads" / (row["file_path"] or "")
            had_local = local.is_file()
            try:
                path = materialize_textbook_file(row["file_path"])
            except Exception as exc:
                print(f"  id={uid}: cannot fetch PDF ({exc})")
                failures += 1
                continue
            try:
                # Same metadata the upload pipeline writes (upload_handler.py).
                extra_meta = {
                    "board": str(row["board"]),
                    "class_level": str(row["class_level"]),
                    "subject_name": row["subject_name"],
                    "textbook_upload_id": str(row["id"]),
                    "content_type": row["content_type"] or "",
                    "content_label": row["content_label"] or "",
                }
                chunks = process_document(path, extra_metadata=extra_meta)
            finally:
                # Don't leave a cached copy behind if the PDF wasn't on disk before.
                if not had_local and path and os.path.isfile(path):
                    try:
                        os.remove(path)
                        os.removedirs(os.path.dirname(path))  # only removes now-empty dirs
                    except OSError:
                        pass
            pages = sorted({int((c.metadata or {}).get("page", 0) or 0) for c in chunks})
            span = f"pages {pages[0]}-{pages[-1]}" if pages else "no pages"
            print(
                f"  id={uid} {row['class_level']} {row['subject_name']} | {row['chapter']}: "
                f"indexed now={before}, extracted={len(chunks)} ({span})"
            )
            if len(chunks) <= before:
                print("    skip: extraction is not larger than what is already indexed")
                continue
            if not args.apply:
                continue
            delete_collection_docs(coll, where_filter={"textbook_upload_id": str(uid)})
            added = 0
            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                added += add_documents_to_store(
                    chunks[i : i + EMBEDDING_BATCH_SIZE],
                    collection_name=coll,
                    replace_existing=False,
                )
            after = _indexed_count(client, coll, uid)
            ok = after == len(chunks)
            failures += 0 if ok else 1
            db.execute(
                text("UPDATE textbook_uploads SET chunk_count=:n WHERE id=:id"),
                {"n": len(chunks), "id": uid},
            )
            db.commit()
            print(f"    {'OK' if ok else 'MISMATCH'}: added={added} indexed after={after}")
        return 1 if failures else 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

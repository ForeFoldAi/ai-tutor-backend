"""Read-only audit: is every uploaded chapter fully present in the vector index?

A chapter whose lesson pages never reached Qdrant still "works" — retrieval returns
the exercises or teacher transcripts that did get indexed, and the tutor confidently
explains the wrong material (Poorvi Class 8 "Wit and Wisdom": only pages 38-47 indexed).

For each textbook_uploads row this compares the DB chunk_count with the number of
points in Qdrant and checks which PDF pages the indexed chunks cover.

Usage (from ai-tutor-backend):
    .venv/bin/python scripts/audit_index_coverage.py              # summary + flagged rows
    .venv/bin/python scripts/audit_index_coverage.py --csv out.csv
    .venv/bin/python scripts/audit_index_coverage.py --board CBSE --class-level CLASS_8

Nothing is written to the database or the index.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--board", default="")
    ap.add_argument("--class-level", default="")
    ap.add_argument("--csv", default="", help="write every row (not just flagged) to this CSV")
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=0.8,
        help="flag when indexed points < this fraction of the DB chunk_count",
    )
    args = ap.parse_args()

    from sqlalchemy import text

    from app.core.database import SessionLocal
    from app.services.chapter_collection import collection_name
    from app.services.vector_backend.qdrant_text import get_qdrant_client, textbook_upload_filter

    sql = (
        "SELECT id, board::text, class_level::text, subject_name, chapter, chunk_count, "
        "embedding_status::text FROM textbook_uploads WHERE 1=1"
    )
    params: dict[str, str] = {}
    if args.board:
        sql += " AND board::text = :board"
        params["board"] = args.board
    if args.class_level:
        sql += " AND class_level::text = :cl"
        params["cl"] = args.class_level
    sql += " ORDER BY board, class_level, subject_name, id"

    with SessionLocal() as db:
        rows = db.execute(text(sql), params).fetchall()

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    out: list[dict] = []
    for uid, board, cl, subject, chapter, chunk_count, emb_status in rows:
        coll = collection_name(board or "", cl or "", subject or "")
        rec = {
            "id": uid,
            "board": board,
            "class_level": cl,
            "subject": subject,
            "chapter": chapter,
            "collection": coll,
            "db_chunks": int(chunk_count or 0),
            "embedding_status": emb_status,
            "indexed": 0,
            "min_page": "",
            "max_page": "",
            "pages_covered": 0,
            "problem": "",
        }
        if coll not in existing:
            rec["problem"] = "collection missing"
            out.append(rec)
            continue
        flt = textbook_upload_filter([str(uid)])
        pages: set[int] = set()
        offset = None
        n = 0
        while True:
            recs, offset = client.scroll(
                collection_name=coll,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=["page"],
                with_vectors=False,
            )
            for r in recs:
                n += 1
                try:
                    pages.add(int((r.payload or {}).get("page", 0) or 0))
                except (TypeError, ValueError):
                    pass
            if offset is None:
                break
        rec["indexed"] = n
        if pages:
            rec["min_page"] = min(pages)
            rec["max_page"] = max(pages)
            rec["pages_covered"] = len(pages)
        problems = []
        if n == 0:
            problems.append("not indexed")
        elif rec["db_chunks"] and n < args.min_ratio * rec["db_chunks"]:
            problems.append(f"only {n}/{rec['db_chunks']} chunks indexed")
        # Lesson text starts on the first pages; an index that begins deep in the
        # chapter is missing the lesson itself.
        if pages and min(pages) > 3:
            problems.append(f"starts at page {min(pages)}")
        if pages and rec["max_page"] and len(pages) < 0.5 * (max(pages) - min(pages) + 1):
            problems.append(f"gaps: {len(pages)} of {max(pages) - min(pages) + 1} pages")
        rec["problem"] = "; ".join(problems)
        out.append(rec)

    flagged = [r for r in out if r["problem"]]
    by_subject: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in out:
        k = (r["board"], r["class_level"], r["subject"])
        by_subject[k][0] += 1
        by_subject[k][1] += 1 if r["problem"] else 0

    print(f"Chapters checked: {len(out)}   flagged: {len(flagged)}")
    print("\nBy book (flagged / total):")
    for (b, cl, s), (total, bad) in sorted(by_subject.items()):
        mark = "  <--" if bad else ""
        print(f"  {b:12} {cl:9} {s[:40]:40} {bad:3} / {total:3}{mark}")
    if flagged:
        print("\nFlagged chapters:")
        for r in flagged:
            print(
                f"  id={r['id']:<5} {r['class_level']:9} {str(r['subject'])[:28]:28} "
                f"{str(r['chapter'])[:40]:40} indexed={r['indexed']:<4} db={r['db_chunks']:<4} "
                f"pages={r['min_page']}-{r['max_page']}  {r['problem']}"
            )
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()) if out else ["id"])
            w.writeheader()
            w.writerows(out)
        print(f"\nWrote {len(out)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

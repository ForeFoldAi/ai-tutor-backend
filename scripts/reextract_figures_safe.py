#!/usr/bin/env python3
"""Re-extract textbook figures with the current pipeline — previewed and gated.

Any board / class / subject. Three modes:

  --preview DIR   Extract offline (no DB or storage writes) and write DIR/review.html:
                  current figures next to the new ones, per chapter. Use it to sign off.
  --apply         For each chapter: extract offline first; only if the new run keeps at
                  least --min-keep of the numbered figures already stored, replace that
                  chapter's figures (old rows/blobs purged, new ones saved and indexed).
                  Then re-run the junk filter across each book and (with --captions)
                  write student captions with the vision model.
  (neither)       Dry run: print the per-chapter comparison only.

  cd ai-tutor-backend && .venv/bin/python scripts/reextract_figures_safe.py --board CBSE --class-level CLASS_10 --subject Mathematics --preview /tmp/fig_preview
  cd ai-tutor-backend && .venv/bin/python scripts/reextract_figures_safe.py --board CBSE --class-level CLASS_10 --apply --captions
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

load_dotenv(BACKEND_ROOT / ".env")

from sqlalchemy import select

from app.core.database import SessionLocal
from app.modules.catalog.models import TextbookImage, TextbookUpload

logger = logging.getLogger("reextract_figures_safe")


def _enum(v) -> str:
    return str(getattr(v, "value", v))


def _uploads(db, args) -> list[TextbookUpload]:
    q = select(TextbookUpload).order_by(TextbookUpload.id)
    if args.upload_id:
        q = q.where(TextbookUpload.id.in_(args.upload_id))
    if args.board:
        q = q.where(TextbookUpload.board == args.board)
    if args.class_level:
        q = q.where(TextbookUpload.class_level == args.class_level)
    if args.subject:
        q = q.where(TextbookUpload.subject_name == args.subject)
    return [u for u in db.scalars(q) if (u.file_path or "").lower().endswith(".pdf")]


def _local_pdf(upload: TextbookUpload, tmp: Path) -> Path | None:
    from app.services.image_service.storage_backend import get_document_storage_backend

    key = (upload.file_path or "").replace("\\", "/")
    if os.path.isfile(key):
        return Path(key)
    data = get_document_storage_backend().get_bytes(key)
    if not data:
        return None
    out = tmp / f"{upload.id}.pdf"
    out.write_bytes(data)
    return out


def _current(db, upload_id: int) -> list[TextbookImage]:
    return list(
        db.scalars(
            select(TextbookImage)
            .where(TextbookImage.textbook_upload_id == upload_id, TextbookImage.content_kind == "figure")
            .order_by(TextbookImage.page_index, TextbookImage.sequence)
        )
    )


def _numbered(figs) -> set[str]:
    return {str(f.figure_number).strip() for f in figs if (f.figure_number or "").strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board")
    ap.add_argument("--class-level")
    ap.add_argument("--subject")
    ap.add_argument("--upload-id", type=int, action="append")
    ap.add_argument("--preview", metavar="DIR")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--captions", action="store_true", help="with --apply: write VLM captions")
    ap.add_argument("--min-keep", type=float, default=0.8,
                    help="new run must keep this share of stored numbered figures (default 0.8)")
    args = ap.parse_args()
    if not (args.board or args.class_level or args.subject or args.upload_id):
        ap.error("pass a filter (--board/--class-level/--subject/--upload-id)")
    if args.preview and args.apply:
        ap.error("--preview and --apply are separate steps")
    logging.basicConfig(level=logging.WARNING)

    from app.services.image_service.pdf_extraction_types import extract_document_layout
    from app.services.image_service.storage_backend import get_storage_backend
    from app.services.image_service.textbook_image_extraction import image_storage_key
    import app.services.image_service.textbook_image_extraction as tie

    # The script runs captions itself (synchronously); don't also start the background thread.
    tie._describe_images_after_extract = lambda upload: None

    preview_dir = Path(args.preview) if args.preview else None
    sections: list[str] = []
    replaced: list[int] = []
    books: set[tuple[str, str, str]] = set()
    backend = get_storage_backend()

    with SessionLocal() as db, tempfile.TemporaryDirectory() as tmpd:
        uploads = _uploads(db, args)
        print(f"{len(uploads)} PDF chapters")
        for up in uploads:
            pdf = _local_pdf(up, Path(tmpd))
            if pdf is None:
                print(f"  upload {up.id}: PDF missing in storage — skipped")
                continue
            old = _current(db, up.id)
            try:
                result = extract_document_layout(str(pdf), max_figures=400, chapter_title=up.chapter)
            except Exception as exc:
                print(f"  upload {up.id}: extraction failed ({exc}) — skipped")
                continue
            new_nums = {str(f.figure_number).strip() for f in result.figures if (f.figure_number or "").strip()}
            old_nums = _numbered(old)
            kept = len(old_nums & new_nums) / len(old_nums) if old_nums else 1.0
            ok = kept >= args.min_keep
            label = f"{_enum(up.board)} {_enum(up.class_level)} {up.subject_name} · {up.chapter}"
            print(f"  upload {up.id} {label[:70]:70} old={len(old)} new={len(result.figures)} "
                  f"numbered old={len(old_nums)} new={len(new_nums)} kept={kept:.0%} {'OK' if ok else 'BLOCKED'}")

            if preview_dir:
                d = preview_dir / str(up.id)
                (d / "new").mkdir(parents=True, exist_ok=True)
                (d / "old").mkdir(parents=True, exist_ok=True)
                cells_new = []
                for k, f in enumerate(result.figures):
                    name = f"{k:03d}.png"
                    (d / "new" / name).write_bytes(f.image_bytes)
                    cells_new.append(
                        f"<figure><img src='{up.id}/new/{name}'><figcaption>p{f.page_index + 1} "
                        f"fig {html.escape(f.figure_number or '—')}<br>{html.escape((f.caption or '')[:140])}"
                        f"</figcaption></figure>"
                    )
                cells_old = []
                for f in old:
                    data = backend.get_bytes(image_storage_key(up, f.file_name))
                    if not data:
                        continue
                    name = f"{f.id}.img"
                    (d / "old" / name).write_bytes(data)
                    hidden = " (hidden)" if f.is_decorative else ""
                    cells_old.append(
                        f"<figure><img src='{up.id}/old/{name}'><figcaption>p{f.page_index + 1} "
                        f"fig {html.escape(f.figure_number or '—')}{hidden}</figcaption></figure>"
                    )
                sections.append(
                    f"<h2>{html.escape(label)} — upload {up.id} — {'OK' if ok else 'BLOCKED'} "
                    f"(numbered kept {kept:.0%})</h2><div class='cols'><div><h3>Current ({len(old)})</h3>"
                    f"{''.join(cells_old)}</div><div><h3>New ({len(result.figures)})</h3>{''.join(cells_new)}</div></div>"
                )
                continue

            if not args.apply:
                continue
            if not ok:
                print(f"    not replaced: new run would lose numbered figures {sorted(old_nums - new_nums)[:12]}")
                continue
            n = tie.reextract_textbook_images(db, up)
            print(f"    replaced: {n} rows")
            if n:
                replaced.append(up.id)
                books.add((_enum(up.board), _enum(up.class_level), up.subject_name))

    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        (preview_dir / "review.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Re-extraction preview</title><style>"
            "body{font:13px system-ui;margin:16px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
            "figure{display:inline-block;width:220px;margin:4px;vertical-align:top}"
            "img{max-width:220px;max-height:200px;object-fit:contain;border:1px solid #ddd}"
            f"</style>{''.join(sections)}",
            encoding="utf-8",
        )
        print(f"Preview: {preview_dir / 'review.html'}")
        return 0

    if args.apply and books:
        flag = BACKEND_ROOT / "scripts" / "flag_junk_figures.py"
        for board, cls, subj in sorted(books):
            print(f"Junk filter: {board} {cls} {subj}")
            subprocess.run([sys.executable, str(flag), "--board", board, "--class-level", cls,
                            "--subject", subj, "--apply"], check=False)
        if args.captions:
            from app.services.image_service.figure_caption_vlm import caption_upload_figures

            for uid in replaced:
                print(f"Captions upload {uid}: {caption_upload_figures(uid)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

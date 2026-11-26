#!/usr/bin/env python3
"""Figure extraction quality harness (Phase 0). Works for any board / class / subject.

Three commands:

  report  DB-only health metrics per book: hidden junk, pHash coverage, what students
          actually see as captions, suspected blind-window crops. Fast, no downloads.

  sample  Stratified review sample (N figures per book) → images + review.html where a
          reviewer marks each figure junk / crop ok / caption ok and downloads labels.jsonl.

  score   Score a labels.jsonl against the current DB: junk precision/recall of the
          ``is_decorative`` flag, crop accuracy, caption accuracy, vs release targets.

  cd ai-tutor-backend && .venv/bin/python scripts/figure_quality.py report --board CBSE --class-level CLASS_10
  cd ai-tutor-backend && .venv/bin/python scripts/figure_quality.py sample --board CBSE --per-book 8 --out /tmp/fq
  cd ai-tutor-backend && .venv/bin/python scripts/figure_quality.py score evaluation/goldens/figure_quality/cbse10_seed.jsonl

Label line format (JSON per line):
  {"image_id": 123, "junk": false, "crop_ok": true, "caption_ok": false, "note": "..."}
``crop_ok`` / ``caption_ok`` may be null (not judged). Junk figures are not judged for
crop or caption.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
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

# Release targets from the extraction plan.
TARGET_JUNK_RATE = 0.01
TARGET_CROP_OK = 0.95
TARGET_CAPTION_OK = 0.90

# _fallback_figure_box: max(220, label_w + 40) wide; stored boxes come out 228–236 px.
_BLIND_WINDOW_WIDTH = range(228, 237)
_BLIP_RE = re.compile(r"(?i)^(a|an|the|there is|there are)\s")
_GENERIC_CAPTIONS = {"textbook illustration", ""}


def _enum(v) -> str:
    return str(getattr(v, "value", v))


def _book(up: TextbookUpload) -> str:
    return f"{_enum(up.board)} / {_enum(up.class_level)} / {up.subject_name}"


def _query_figures(db, args, *, include_hidden: bool = True) -> list[TextbookImage]:
    q = (
        select(TextbookImage)
        .join(TextbookUpload, TextbookUpload.id == TextbookImage.textbook_upload_id)
        .where(TextbookImage.content_kind == "figure")
        .options(selectinload(TextbookImage.upload))
        .order_by(TextbookImage.id)
    )
    if getattr(args, "board", None):
        q = q.where(TextbookUpload.board == args.board)
    if getattr(args, "class_level", None):
        q = q.where(TextbookUpload.class_level == args.class_level)
    if getattr(args, "subject", None):
        q = q.where(TextbookUpload.subject_name == args.subject)
    if not include_hidden:
        q = q.where(TextbookImage.is_decorative.is_(False))
    return list(db.scalars(q))


def _display_caption(im: TextbookImage) -> str:
    from app.services.image_service.pdf_figure_context import resolve_display_caption

    try:
        return resolve_display_caption(im)
    except Exception as exc:  # report, never crash the harness
        return f"<error: {exc.__class__.__name__}>"


def _caption_bucket(text: str, im: TextbookImage) -> str:
    t = (text or "").strip()
    if t.lower() in _GENERIC_CAPTIONS:
        return "generic"
    if "textbook figure" in t.lower() or (im.caption_source == "vision" and _BLIP_RE.match(t)):
        return "blip_style"
    if len(t) < 12:
        return "fragment"
    return "text"


def _blind_window(im: TextbookImage) -> bool:
    try:
        x0, _, x1, _ = json.loads(im.image_bbox or "[]")
    except Exception:
        return False
    return round(x1 - x0) in _BLIND_WINDOW_WIDTH


# ── report ──────────────────────────────────────────────────────────────────────

def cmd_report(args) -> int:
    logging.disable(logging.WARNING)
    db = SessionLocal()
    try:
        figs = _query_figures(db, args)
    finally:
        db.close()
    books: dict[str, Counter] = defaultdict(Counter)
    for im in figs:
        c = books[_book(im.upload)]
        c["figures"] += 1
        if im.is_decorative:
            c["hidden"] += 1
            continue
        c["shown"] += 1
        c["phash"] += im.phash is not None
        c["fig_no"] += bool((im.figure_number or "").strip())
        c["blind_window"] += _blind_window(im)
        c["vlm_caption"] += im.caption_source == "vlm"
        c["needs_recrop"] += "needs_recrop" in (im.educational_tags or "")
        c["cap_" + _caption_bucket(_display_caption(im), im)] += 1

    cols = ["figures", "hidden", "shown", "phash", "fig_no", "blind_window", "needs_recrop",
            "vlm_caption", "cap_text", "cap_blip_style", "cap_generic", "cap_fragment"]
    print(f"{'book':58} " + " ".join(f"{c[:12]:>12}" for c in cols))
    total = Counter()
    for name in sorted(books):
        total.update(books[name])
        print(f"{name[:58]:58} " + " ".join(f"{books[name][c]:>12}" for c in cols))
    print(f"{'TOTAL':58} " + " ".join(f"{total[c]:>12}" for c in cols))
    shown = max(1, total["shown"])
    print(
        f"\nOf shown figures: weak captions (blip/generic/fragment) "
        f"{(total['cap_blip_style'] + total['cap_generic'] + total['cap_fragment']) / shown:.1%}, "
        f"suspected blind-window crops {total['blind_window'] / shown:.1%}, "
        f"pHash coverage {total['phash'] / shown:.1%}, "
        f"VLM captions {total['vlm_caption'] / shown:.1%}"
    )
    if args.json:
        Path(args.json).write_text(json.dumps({k: dict(v) for k, v in books.items()}, indent=1))
        print(f"JSON: {args.json}")
    return 0


# ── sample ──────────────────────────────────────────────────────────────────────

_REVIEW_JS = """
function collect(){const out=[];document.querySelectorAll('.card').forEach(c=>{
 const g=n=>{const v=c.querySelector('input[name="'+n+c.dataset.id+'"]:checked');return v?JSON.parse(v.value):null};
 out.push(JSON.stringify({image_id:+c.dataset.id,junk:g('junk'),crop_ok:g('crop'),caption_ok:g('cap'),
  note:c.querySelector('textarea').value}))});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.join('\\n')+'\\n'],{type:'application/x-ndjson'}));
 a.download='labels.jsonl';a.click();}
"""


def _radio(name: str, img_id: int) -> str:
    return " ".join(
        f'<label><input type="radio" name="{name}{img_id}" value="{v}">{t}</label>'
        for v, t in (("true", "yes"), ("false", "no"))
    )


def cmd_sample(args) -> int:
    from app.services.image_service.storage_backend import get_storage_backend
    from app.services.image_service.textbook_image_extraction import image_storage_key

    logging.disable(logging.WARNING)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    try:
        figs = _query_figures(db, args, include_hidden=args.include_hidden)
        by_book: dict[str, list[TextbookImage]] = defaultdict(list)
        for im in figs:
            by_book[_book(im.upload)].append(im)
        rng = random.Random(args.seed)
        picked = [im for name in sorted(by_book)
                  for im in rng.sample(by_book[name], min(args.per_book, len(by_book[name])))]
        captions = {im.id: _display_caption(im) for im in picked}
        backend = get_storage_backend()

        def fetch(im: TextbookImage) -> bool:
            data = backend.get_bytes(image_storage_key(im.upload, im.file_name))
            if data:
                (out / "img" / f"{im.id}.jpg").write_bytes(data)
            return bool(data)

        with ThreadPoolExecutor(12) as ex:
            ok = dict(zip([im.id for im in picked], ex.map(fetch, picked)))
    finally:
        db.close()

    cards = []
    for im in picked:
        if not ok.get(im.id):
            continue
        meta = (f"#{im.id} · {html.escape(_book(im.upload))} · {html.escape(im.upload.chapter or '')} · "
                f"page {im.page_index + 1} · fig {html.escape(im.figure_number or '—')}"
                f"{' · HIDDEN' if im.is_decorative else ''}")
        cards.append(
            f'<div class="card" data-id="{im.id}"><img src="img/{im.id}.jpg"><div class="meta">{meta}</div>'
            f'<div class="cap">Student sees: <b>{html.escape(captions[im.id])}</b></div>'
            f'<div>Junk (QR, banner, mascot, decoration, not a teaching figure)? {_radio("junk", im.id)}</div>'
            f'<div>Crop shows the whole figure and nothing else? {_radio("crop", im.id)}</div>'
            f'<div>Caption correct and clear for this class? {_radio("cap", im.id)}</div>'
            f'<textarea placeholder="note"></textarea></div>'
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Figure review</title><style>"
        "body{font:14px system-ui;margin:16px}.card{border:1px solid #ccc;padding:8px;margin:8px 0;"
        "display:grid;gap:4px}img{max-width:480px;max-height:360px;object-fit:contain;background:#fafafa}"
        ".meta{color:#666}textarea{width:480px;height:32px}button{position:sticky;top:0;padding:8px 16px}"
        f"</style><button onclick='collect()'>Download labels.jsonl</button>{''.join(cards)}"
        f"<script>{_REVIEW_JS}</script>"
    )
    (out / "review.html").write_text(page, encoding="utf-8")
    print(f"{len(cards)} figures from {len(by_book)} books → {out / 'review.html'}")
    return 0


# ── score ───────────────────────────────────────────────────────────────────────

def _rate(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.1%}" if d else "n/a"


def cmd_score(args) -> int:
    logging.disable(logging.WARNING)
    labels = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    db = SessionLocal()
    try:
        rows = {
            im.id: im
            for im in db.scalars(
                select(TextbookImage)
                .where(TextbookImage.id.in_([int(l["image_id"]) for l in labels]))
                .options(selectinload(TextbookImage.upload))
            )
        }
    finally:
        db.close()

    tp = fp = fn = 0
    shown_junk = shown = 0
    crop = Counter()
    cap = Counter()
    per_book: dict[str, Counter] = defaultdict(Counter)
    for lab in labels:
        im = rows.get(int(lab["image_id"]))
        if im is None:
            continue
        book = per_book[_book(im.upload)]
        junk = lab.get("junk")
        hidden = bool(im.is_decorative)
        if junk is True:
            tp += hidden
            fn += not hidden
        elif junk is False:
            fp += hidden
        if not hidden:
            shown += 1
            shown_junk += junk is True
            book["shown"] += 1
            book["shown_junk"] += junk is True
        if junk is True:
            continue
        for key, ctr in (("crop_ok", crop), ("caption_ok", cap)):
            v = lab.get(key)
            if v is None:
                continue
            ctr["n"] += 1
            ctr["ok"] += bool(v)
            book[key + "_n"] += 1
            book[key] += bool(v)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    junk_rate = shown_junk / shown if shown else 0.0
    crop_rate = crop["ok"] / crop["n"] if crop["n"] else 0.0
    cap_rate = cap["ok"] / cap["n"] if cap["n"] else 0.0
    print(f"Labels: {len(labels)} (matched {len(rows)})")
    print(f"Junk filter  precision {precision:.1%}  recall {recall:.1%}  (tp={tp} fp={fp} fn={fn})")
    print(f"Junk shown to students  {_rate(shown_junk, shown)}   target ≤ {TARGET_JUNK_RATE:.0%}  "
          f"{'PASS' if junk_rate <= TARGET_JUNK_RATE else 'FAIL'}")
    print(f"Crop ok                 {_rate(crop['ok'], crop['n'])}   target ≥ {TARGET_CROP_OK:.0%}  "
          f"{'PASS' if crop['n'] and crop_rate >= TARGET_CROP_OK else 'FAIL'}")
    print(f"Caption ok              {_rate(cap['ok'], cap['n'])}   target ≥ {TARGET_CAPTION_OK:.0%}  "
          f"{'PASS' if cap['n'] and cap_rate >= TARGET_CAPTION_OK else 'FAIL'}")
    print("\nPer book (shown junk · crop ok · caption ok):")
    for name in sorted(per_book):
        b = per_book[name]
        print(f"  {name[:56]:56} {_rate(b['shown_junk'], b['shown']):>16} "
              f"{_rate(b['crop_ok'], b['crop_ok_n']):>16} {_rate(b['caption_ok'], b['caption_ok_n']):>16}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "sample"):
        p = sub.add_parser(name)
        p.add_argument("--board")
        p.add_argument("--class-level")
        p.add_argument("--subject")
    sub.choices["report"].add_argument("--json", help="also write per-book metrics here")
    s = sub.choices["sample"]
    s.add_argument("--per-book", type=int, default=8)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", required=True)
    s.add_argument("--include-hidden", action="store_true", help="also sample is_decorative rows")
    sc = sub.add_parser("score")
    sc.add_argument("labels")
    args = ap.parse_args()
    return {"report": cmd_report, "sample": cmd_sample, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

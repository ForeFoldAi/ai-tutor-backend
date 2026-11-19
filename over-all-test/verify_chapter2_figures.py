#!/usr/bin/env python3
"""
Verify chapter-2.pdf figure extraction the same way we debug in chat.

Usage (from ai-tutor-backend):
  PYTHONPATH=. .venv/bin/python over-all-test/verify_chapter2_figures.py
  PYTHONPATH=. .venv/bin/python over-all-test/verify_chapter2_figures.py --write-crops

Writes a report to over-all-test/chapter-2-figures/report.json
and optional JPEG crops under over-all-test/chapter-2-figures/crops/.
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

os.environ.setdefault("PDF_EXTRACTION_PIPELINE_ENABLED", "true")

EXPECTED_NUMBERS = [
    "2.1",
    "2.2",
    "2.3.1",
    "2.3.2",
    "2.3.3",
    "2.4.1",
    "2.4.2",
    "2.4.3",
    "2.4.4",
    "2.5",
    "2.6",
    "2.7",
    "2.8",
    "2.9",
    "2.10",
    "2.11",
    "2.12",
    "2.13",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify chapter-2 figure extraction")
    parser.add_argument(
        "--pdf",
        default=str(Path(__file__).resolve().parent / "chapter-2.pdf"),
        help="Path to chapter-2.pdf",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "chapter-2-figures"),
        help="Report / crop output directory",
    )
    parser.add_argument(
        "--write-crops",
        action="store_true",
        help="Write figure JPEG crops next to the report",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    from app.services.pdf_extract_pipeline.cache import clear_pipeline_cache
    from app.services.pdf_extract_pipeline.bridge import extract_with_ml_pipeline

    clear_pipeline_cache()
    doc, _, _ = extract_with_ml_pipeline(str(pdf_path))

    figures = sorted(
        [f for f in doc.figures if (f.source_type or "").startswith("ml_figure") or True],
        key=lambda f: (f.page_index, f.figure_number or "z", f.sequence),
    )
    # Only keep figure-like assets (exclude tables/formulas if mixed).
    figures = [f for f in figures if (getattr(f, "source_type", "") or "").endswith("figure")
               or getattr(f, "source_type", "") in ("", "ml_figure")
               or f.figure_number]

    rows = []
    numbered = []
    orphans = []
    for f in figures:
        bb = f.image_bbox
        w = int(bb.x1 - bb.x0) if bb else 0
        h = int(bb.y1 - bb.y0) if bb else 0
        row = {
            "page_index": f.page_index,
            "figure_number": f.figure_number,
            "caption": (f.caption or "")[:240],
            "size": f"{w}x{h}",
            "width": w,
            "height": h,
            "bytes": len(f.image_bytes or b""),
            "source_type": getattr(f, "source_type", None),
        }
        rows.append(row)
        if f.figure_number:
            numbered.append(f.figure_number)
        else:
            orphans.append(row)
        print(
            f"p{f.page_index:02d} fig={str(f.figure_number):6s} "
            f"{w}x{h}  {(f.caption or '')[:80]}"
        )

    missing = [n for n in EXPECTED_NUMBERS if n not in numbered]
    extra = [n for n in numbered if n not in EXPECTED_NUMBERS]
    report = {
        "pdf": str(pdf_path),
        "figure_count": len(rows),
        "numbered": numbered,
        "expected": EXPECTED_NUMBERS,
        "missing": missing,
        "extra_numbers": extra,
        "orphan_count": len(orphans),
        "ok": not missing and not orphans and not extra,
        "figures": rows,
        "caption_notes": {
            "source": "Paired from PDF Fig. label + caption line during ML layout pairing "
            "(figure_pairing._caption_for_numbered_figure). "
            "After DB persist, enrich may add generated_caption (nearby-text heuristic) "
            "or vision (BLIP) when OCR caption is weak/corrupt; "
            "resolve_display_caption picks the student-facing string.",
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    print(f"figures={len(rows)} missing={missing} orphans={len(orphans)} ok={report['ok']}")

    if args.write_crops:
        crop_dir = out_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for f in figures:
            num = (f.figure_number or f"p{f.page_index}_{f.sequence}").replace(".", "_")
            name = f"fig_{num}.jpg" if f.figure_number else f"orphan_p{f.page_index}_{f.sequence}.jpg"
            data = f.image_bytes or b""
            if not data:
                continue
            # Normalize to JPEG when PNG bytes come from pipeline.
            out = crop_dir / name
            if data[:2] == b"\xff\xd8":
                out.write_bytes(data)
            else:
                try:
                    from io import BytesIO
                    from PIL import Image

                    im = Image.open(BytesIO(data)).convert("RGB")
                    im.save(out, format="JPEG", quality=88)
                except Exception:
                    out.with_suffix(".bin").write_bytes(data)
        print(f"Wrote crops to {crop_dir}")

    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

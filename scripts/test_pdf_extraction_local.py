#!/usr/bin/env python3
"""Run ML-only PDF extraction on a local file (no DB). Usage: python scripts/test_pdf_extraction_local.py gees102.pdf"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PDF_EXTRACTION_PIPELINE_ENABLED", "true")


def main() -> int:
    pdf_name = (sys.argv[1] if len(sys.argv) > 1 else "gees102.pdf").strip()
    pdf_path = Path(pdf_name)
    if not pdf_path.is_file():
        pdf_path = ROOT / pdf_name
    if not pdf_path.is_file():
        print(f"FAIL: PDF not found: {pdf_name}")
        return 2

    from app.services.pdf_extract_pipeline.config import pipeline_enabled, load_pipeline_config
    from app.services.pdf_extract_pipeline.bridge import (
        MlPipelineExtractionError,
        extract_with_ml_pipeline,
    )

    print("PDF:", pdf_path)
    print("Size:", pdf_path.stat().st_size, "bytes")
    print("ML pipeline enabled:", pipeline_enabled())
    cfg = load_pipeline_config()
    print("Stages:", json.dumps(cfg.get("stages", {}), indent=2))

    out_dir = ROOT / "uploads" / "_extraction_test" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    formulas_dir = out_dir / "formulas"
    for d in (figures_dir, tables_dir, formulas_dir):
        d.mkdir(exist_ok=True)

    print("\n=== ML-only extraction (no caption-anchored fallback) ===")
    try:
        result, ml_assets, pipeline = extract_with_ml_pipeline(
            str(pdf_path),
            max_figures=96,
            chapter_title=pdf_path.stem,
        )
    except MlPipelineExtractionError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"Figures: {len(result.figures)}")
    print(f"Table/formula assets: {len(ml_assets)}")
    print(f"Pipeline pages: {len(pipeline.pages)}")
    print(f"Pipeline assets (all): {len(pipeline.assets)}")

    for i, a in enumerate(pipeline.assets):
        if i >= 25:
            print(f"  ... and {len(pipeline.assets) - 25} more assets")
            break
        print(f"  [{a.asset_type}] page={a.page_no} num={a.number or '-'} caption={a.caption[:60]!r}")

    for pf in result.figures:
        fname = pf.preferred_file_name or f"p{pf.page_index}_{pf.sequence}.jpg"
        dest = figures_dir / fname
        with open(dest, "wb") as f:
            f.write(pf.image_bytes)
        print(f"  saved figure: {fname} (Fig {pf.figure_number}, page {pf.page_number})")

    for asset in ml_assets:
        if asset.content_kind == "table":
            dest = tables_dir / f"table_p{asset.page_index}_{asset.sequence}.jpg"
        else:
            dest = formulas_dir / f"formula_p{asset.page_index}_{asset.sequence}.jpg"
        with open(dest, "wb") as f:
            f.write(asset.image_bytes)
        print(f"  saved {asset.content_kind}: {dest.name}")

    text_path = out_dir / "extracted_text.md"
    text_path.write_text(pipeline.full_text[:50000], encoding="utf-8")
    print(f"\nText ({len(pipeline.full_text)} chars) → {text_path}")
    if pipeline.full_text.strip():
        print(pipeline.full_text[:500])
    print(f"\nOutput dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

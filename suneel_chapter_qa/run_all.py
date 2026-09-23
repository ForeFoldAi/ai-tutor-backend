#!/usr/bin/env python3
"""Run all chapter banks, then write results/index.md.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/run_all.py
  .venv/bin/python suneel_chapter_qa/run_all.py --only english_unit1,math_ch1_square_cube

After each chapter finishes, type yes to continue (use --no-prompt to skip).
Each question gets one follow-up immediately; there is no end-of-chapter FU batch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BANKS = ROOT / "banks"
RESULTS = ROOT / "results"
RAW = ROOT / "raw"
PY = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")


def wait_for_yes(slug: str, subject: str, chapter: str, *, remaining: int) -> None:
    """Block until the user types yes to start the next chapter."""
    while True:
        ans = input(
            f"\nFinished {subject} — {chapter} ({slug}). "
            f"{remaining} left. Type yes to continue (or quit to stop): "
        ).strip().lower()
        if ans == "yes":
            return
        if ans in ("quit", "q", "n", "no", "stop"):
            raise SystemExit("stopped by user")
        print("Enter yes to continue, or quit to stop.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated slugs")
    ap.add_argument("--channel", default="both", choices=("tutor", "voice", "both"))
    ap.add_argument("--skip-done", action="store_true", help="skip slug if results/<slug>.md exists")
    ap.add_argument("--no-prompt", action="store_true", help="do not pause between chapters")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from lib.api import require_api

    # Fail once up front — every chapter needs POST /auth/login on :8000.
    require_api()

    index = json.loads((BANKS / "_index.json").read_text())
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows = [r for r in index if not only or r["slug"] in only]
    slugs = [r["slug"] for r in rows]
    by_slug = {r["slug"]: r for r in rows}

    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    failed = 0
    for i, slug in enumerate(slugs):
        md = RESULTS / f"{slug}.md"
        if args.skip_done and md.exists():
            print(f"skip {slug} (exists)", flush=True)
            continue
        print(f"\n######## RUN {slug} ########\n", flush=True)
        rc = subprocess.call(
            [PY, str(ROOT / "run_chapter.py"), "--chapter", slug, "--channel", args.channel],
            cwd=str(ROOT.parent),
        )
        if rc != 0:
            failed += 1
            print(f"FAILED {slug} rc={rc}", flush=True)

        # Pause between chapters so you can review before the next subject/chapter.
        remaining = len(slugs) - i - 1
        if remaining > 0 and not args.no_prompt:
            meta = by_slug[slug]
            wait_for_yes(
                slug,
                meta.get("subject") or "?",
                meta.get("chapter") or slug,
                remaining=remaining,
            )

    # rebuild index from raw json
    summaries = []
    for row in index:
        slug = row["slug"]
        raw = RAW / f"{slug}.json"
        if not raw.exists():
            continue
        data = json.loads(raw.read_text())
        summaries.append(
            {
                "slug": slug,
                "subject": row["subject"],
                "chapter": row["chapter"],
                "tutor_avg": data.get("tutor_avg", 0),
                "voice_avg": data.get("voice_avg", 0),
            }
        )

    from lib.report import write_index

    write_index(RESULTS, summaries)
    print("Wrote", RESULTS / "index.md")
    print("done chapters", len(summaries), "/", len(slugs))
    if failed:
        raise SystemExit(f"{failed} chapter run(s) failed")


if __name__ == "__main__":
    main()

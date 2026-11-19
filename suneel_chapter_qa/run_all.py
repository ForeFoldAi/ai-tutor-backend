#!/usr/bin/env python3
"""Run all chapter banks, then write results/index.md.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/run_all.py
  .venv/bin/python suneel_chapter_qa/run_all.py --only english_unit1,math_ch1_square_cube
  .venv/bin/python suneel_chapter_qa/run_all.py --channel all --tag after_fixes --no-prompt

After each chapter finishes, type yes to continue (use --no-prompt to skip).
Each question gets one follow-up immediately; there is no end-of-chapter FU batch.
Use --index-only to rebuild index.md from existing raw JSON without calling the API.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BANKS = ROOT / "banks"
PY = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable

sys.path.insert(0, str(ROOT))
from run_chapter import CHANNEL_SETS, out_dirs  # noqa: E402


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
    ap.add_argument("--channel", default="both", choices=tuple(CHANNEL_SETS))
    ap.add_argument("--assistant-mode", choices=("ask", "free", "explain", "practice"), default="ask")
    ap.add_argument("--tag", default="", help="write to runs/<tag>/ instead of results/ + raw/")
    ap.add_argument("--skip-done", action="store_true", help="skip slug if <results>/<slug>.md exists")
    ap.add_argument("--no-prompt", action="store_true", help="do not pause between chapters")
    ap.add_argument("--index-only", action="store_true", help="only rebuild index.md from raw JSON")
    args = ap.parse_args()

    from qa_lib.api import require_api

    results_dir, raw_dir = out_dirs(args.tag)
    index = json.loads((BANKS / "_index.json").read_text())
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows = [r for r in index if not only or r["slug"] in only]
    slugs = [] if args.index_only else [r["slug"] for r in rows]
    by_slug = {r["slug"]: r for r in rows}

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Fail once up front — every chapter needs POST /auth/login on :8000.
    if slugs:
        require_api()

    failed = 0
    for i, slug in enumerate(slugs):
        md = results_dir / f"{slug}.md"
        if args.skip_done and md.exists():
            print(f"skip {slug} (exists)", flush=True)
            continue
        print(f"\n######## RUN {slug} ########\n", flush=True)
        cmd = [
            PY, str(ROOT / "run_chapter.py"), "--chapter", slug,
            "--channel", args.channel, "--assistant-mode", args.assistant_mode,
        ]
        if args.tag:
            cmd += ["--tag", args.tag]
        rc = subprocess.call(cmd, cwd=str(ROOT.parent))
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

    from qa_lib.report import load_summaries, write_index

    summaries = load_summaries(raw_dir, index)
    write_index(results_dir, summaries)
    print("Wrote", results_dir / "index.md")
    print("done chapters", len(summaries), "/", len(slugs))
    if failed:
        raise SystemExit(f"{failed} chapter run(s) failed")


if __name__ == "__main__":
    main()

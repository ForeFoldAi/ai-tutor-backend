#!/usr/bin/env python3
"""Compare two benchmark runs (e.g. baseline vs after fixes, or ministral-8b vs mistral-small).

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/compare.py --a baseline --b after_fixes
  .venv/bin/python suneel_chapter_qa/compare.py --a model_ministral8b --b model_mistral_small \
      --price-a 0.10,0.10 --price-b 0.10,0.30

"baseline" means the top-level raw/ folder; anything else is runs/<tag>/raw.
--price-X is "input,output" USD per million tokens, used for a rough cost estimate
(tokens are approximated as characters / 4 of question + answer; context is not counted).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qa_lib.report import CHANNELS, _sec, latency_stats, load_summaries  # noqa: E402


def raw_dir(tag: str) -> Path:
    return ROOT / "raw" if tag in ("", "baseline") else ROOT / "runs" / tag / "raw"


def _pooled(tag: str) -> dict[str, list[dict]]:
    index = json.loads((ROOT / "banks" / "_index.json").read_text())
    pooled: dict[str, list[dict]] = {k: [] for k, _ in CHANNELS}
    for s in load_summaries(raw_dir(tag), index):
        for k, rows in s["rows"].items():
            pooled[k].extend(rows)
    return pooled


def _tokens(rows: list[dict]) -> tuple[int, int]:
    tin = sum(len(r.get("query") or "") for r in rows) // 4
    tout = sum(len(r.get("answer") or r.get("answer_clip") or "") for r in rows) // 4
    return tin, tout


def _price(s: str) -> tuple[float, float] | None:
    if not s:
        return None
    a, b = s.split(",")
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", required=True)
    ap.add_argument("--price-a", default="")
    ap.add_argument("--price-b", default="")
    args = ap.parse_args()

    pa, pb = _pooled(args.a), _pooled(args.b)
    lines = [
        f"# Benchmark comparison: `{args.a}` vs `{args.b}`",
        "",
        "| Channel | Run | Turns | Avg score | ≥8 | <5 | Median total | p90 total | Median first token |",
        "|---------|-----|-------|-----------|----|----|--------------|-----------|--------------------|",
    ]
    for key, title in CHANNELS:
        for tag, pooled in ((args.a, pa), (args.b, pb)):
            rows = pooled[key]
            if not rows:
                continue
            st = latency_stats(rows)
            lines.append(
                f"| {title} | {tag} | {st['n']} | {st['avg']} | {st['ge8_pct']}% | {st['poor_pct']}% | "
                f"{_sec(st['median_ms'])} | {_sec(st['p90_ms'])} | {_sec(st['ttft_median_ms'])} |"
            )

    prices = {args.a: _price(args.price_a), args.b: _price(args.price_b)}
    if any(prices.values()):
        lines += ["", "## Rough cost (question + answer text only)", ""]
        for tag, pooled in ((args.a, pa), (args.b, pb)):
            price = prices[tag]
            if not price:
                continue
            rows = [r for rs in pooled.values() for r in rs]
            tin, tout = _tokens(rows)
            usd = tin / 1e6 * price[0] + tout / 1e6 * price[1]
            per_1k = usd / max(1, len(rows)) * 1000
            lines.append(f"- `{tag}`: ~{tin + tout:,} tokens, ~${usd:.4f} total, ~${per_1k:.2f} per 1,000 answers")
        lines.append(
            "- Real cost is higher: every call also sends the system prompt and textbook context "
            "(often 2–6k tokens). Use the Mistral console usage page for exact numbers."
        )

    out = ROOT / "runs" / f"compare_{args.a}_vs_{args.b}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote", out)


if __name__ == "__main__":
    main()

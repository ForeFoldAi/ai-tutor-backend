#!/usr/bin/env python3
"""Run one chapter bank on AI Tutor, AI Voice and/or Ask Learn Mastro.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/run_chapter.py --chapter english_unit1
  .venv/bin/python suneel_chapter_qa/run_chapter.py --chapter 26 --channel all --tag after_fixes

Without --tag, output goes to results/ and raw/ (the 2026-09-15 baseline lives there).
With --tag NAME, output goes to runs/NAME/results and runs/NAME/raw.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qa_lib.api import (  # noqa: E402
    AuthSession,
    chat_assistant,
    chat_tutor,
    chat_voice,
    clip,
    img_brief,
    probe_nest,
    require_api,
)
from qa_lib.followups import followup_from_answer  # noqa: E402
from qa_lib.report import latency_stats, slim_for_raw, write_chapter_md  # noqa: E402
from qa_lib.score import score_turn  # noqa: E402

BANKS = ROOT / "banks"
CHANNEL_SETS = {
    "tutor": ("tutor",),
    "voice": ("voice",),
    "assistant": ("assistant",),
    "both": ("tutor", "voice"),
    "all": ("tutor", "voice", "assistant"),
}


def out_dirs(tag: str) -> tuple[Path, Path]:
    if not tag:
        return ROOT / "results", ROOT / "raw"
    base = ROOT / "runs" / tag
    return base / "results", base / "raw"


def load_bank(chapter: str) -> dict:
    index = json.loads((BANKS / "_index.json").read_text())
    slug = None
    for row in index:
        if chapter == row["slug"] or chapter == str(row["chapter_id"]):
            slug = row["slug"]
            break
    if not slug:
        p = BANKS / f"{chapter}.json"
        if p.exists():
            return json.loads(p.read_text())
        raise SystemExit(f"unknown chapter {chapter!r}. known: {[r['slug'] for r in index]}")
    return json.loads((BANKS / f"{slug}.json").read_text())


def score_row(q: dict, res: dict, channel: str) -> dict:
    return score_turn(
        answer=res.get("answer") or "",
        images=res.get("related_images"),
        math_lesson=res.get("math_lesson"),
        science_experiment=res.get("science_experiment"),
        expect_keywords=q.get("expect_keywords") or [],
        forbidden=q.get("forbidden") or [],
        label=q.get("label") or "normal",
        want_images=bool(q.get("want_images")),
        want_math=bool(q.get("want_math")),
        want_science=bool(q.get("want_science")),
        channel=channel,
    )


def pack_result(res: dict) -> dict:
    return {
        "answer": res.get("answer") or "",
        "answer_clip": clip(res.get("answer"), 450),
        "images": img_brief(res.get("related_images")),
        "n_images": len(res.get("related_images") or []),
        "has_math_lesson": bool(res.get("math_lesson")),
        "has_science_experiment": bool(res.get("science_experiment")),
        "ms": res.get("ms"),
        "ttft_ms": res.get("ttft_ms"),
        "error": res.get("error"),
        "path": res.get("path"),
    }


def runner_for(channel: str, assistant_mode: str):
    if channel == "tutor":
        return chat_tutor
    if channel == "voice":
        return chat_voice
    return functools.partial(chat_assistant, agent_mode=assistant_mode)


def run_channel(session: AuthSession, bank: dict, channel: str, *, assistant_mode: str = "ask") -> dict:
    runner = runner_for(channel, assistant_mode)
    meta_kw = dict(
        board=bank["board"],
        class_level=bank["class_level"],
        subject=bank["subject"],
        chapter_id=bank["chapter_id"],
        chapter=bank["chapter"],
    )
    primaries: list[dict] = []
    followups: list[dict] = []
    qs = bank["questions"]
    n = len(qs)
    for i, q in enumerate(qs):
        print(f"  [{channel}] Q {i+1}/{n} {q['id']}…", flush=True)
        res = runner(session, query=q["query"], history=None, **meta_kw)
        primary = {
            "id": q["id"],
            "query": q["query"],
            "label": q["label"],
            "expect_keywords": q.get("expect_keywords") or [],
            **pack_result(res),
            "scores": score_row(q, res, channel),
        }
        primaries.append(primary)
        print(
            f"  [{channel}] {q['id']} -> {primary['scores']['overall']} "
            f"({primary['ms']} ms, first token {primary['ttft_ms']} ms)",
            flush=True,
        )

        # One follow-up right after each primary — no extra FU batch after the chapter ends.
        fu_text, derived = followup_from_answer(primary["answer"], i, q.get("label") or "normal")
        history = [
            {"role": "user", "content": q["query"]},
            {"role": "assistant", "content": primary["answer"]},
        ]
        print(f"  [{channel}] FU {i+1}/{n} ← {q['id']}: {fu_text!r}", flush=True)
        fu_res = runner(session, query=fu_text, history=history, **meta_kw)
        sc = score_row(q, fu_res, channel)
        followups.append(
            {
                "source_id": q["id"],
                "source_query": q["query"],
                "query": fu_text,
                "derived": derived,
                **pack_result(fu_res),
                "scores": sc,
            }
        )
        print(f"  [{channel}] FU {i+1} -> {sc['overall']} derived={derived}", flush=True)

    return {"primaries": primaries, "followups": followups}


def avg_channel(data: dict) -> float:
    scores = [r["scores"]["overall"] for r in data["primaries"]]
    scores += [r["scores"]["overall"] for r in data["followups"]]
    return round(sum(scores) / max(1, len(scores)), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True, help="slug or chapter_id")
    ap.add_argument("--channel", choices=tuple(CHANNEL_SETS), default="both")
    ap.add_argument("--assistant-mode", choices=("ask", "free", "explain", "practice"), default="ask")
    ap.add_argument("--tag", default="", help="write to runs/<tag>/ instead of results/ + raw/")
    args = ap.parse_args()

    results_dir, raw_dir = out_dirs(args.tag)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(args.chapter)
    print(f"=== {bank['slug']} / {bank['chapter']} ===", flush=True)
    require_api()
    session = AuthSession()
    channels = CHANNEL_SETS[args.channel]
    nest = probe_nest(session, bank) if "voice" in channels else {"ok": None, "note": "voice not run"}
    print("nest_probe", nest.get("ok"), nest.get("error"), flush=True)

    data: dict[str, dict] = {}
    for ch in ("tutor", "voice", "assistant"):
        if ch in channels:
            print(ch.upper(), flush=True)
            data[ch] = run_channel(session, bank, ch, assistant_mode=args.assistant_mode)
        else:
            data[ch] = {"primaries": [], "followups": []}

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "meta": {
            "slug": bank["slug"],
            "subject": bank["subject"],
            "chapter_id": bank["chapter_id"],
            "chapter": bank["chapter"],
            "board": bank["board"],
            "class_level": bank["class_level"],
            "questions": bank["questions"],
            "assistant_mode": args.assistant_mode,
            "tag": args.tag,
        },
        "nest_probe": nest,
        **data,
    }
    for ch in ("tutor", "voice", "assistant"):
        payload[f"{ch}_avg"] = avg_channel(data[ch]) if data[ch]["primaries"] else 0.0
        rows = data[ch]["primaries"] + data[ch]["followups"]
        payload[f"{ch}_latency"] = latency_stats(rows) if rows else None

    md_path = results_dir / f"{bank['slug']}.md"
    write_chapter_md(md_path, payload)
    raw_path = raw_dir / f"{bank['slug']}.json"
    raw_path.write_text(json.dumps(slim_for_raw(payload), indent=2))
    print("Wrote", md_path)
    print("Wrote", raw_path)
    print(" ".join(f"{ch.upper()} {payload[f'{ch}_avg']}" for ch in channels))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one chapter bank on AI Tutor and/or AI Voice.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/run_chapter.py --chapter english_unit1
  .venv/bin/python suneel_chapter_qa/run_chapter.py --chapter 26 --channel both
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.api import (  # noqa: E402
    AuthSession,
    chat_tutor,
    chat_voice,
    clip,
    img_brief,
    probe_nest,
    require_api,
)
from lib.followups import followup_from_answer  # noqa: E402
from lib.report import slim_for_raw, write_chapter_md  # noqa: E402
from lib.score import score_turn  # noqa: E402

BANKS = ROOT / "banks"
RESULTS = ROOT / "results"
RAW = ROOT / "raw"


def load_bank(chapter: str) -> dict:
    # slug or chapter_id
    index = json.loads((BANKS / "_index.json").read_text())
    slug = None
    for row in index:
        if chapter == row["slug"] or chapter == str(row["chapter_id"]):
            slug = row["slug"]
            break
    if not slug:
        # direct file
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


def pack_primary(q: dict, res: dict, channel: str) -> dict:
    return {
        "id": q["id"],
        "query": q["query"],
        "label": q["label"],
        "expect_keywords": q.get("expect_keywords") or [],
        "answer": res.get("answer") or "",
        "answer_clip": clip(res.get("answer"), 450),
        "images": img_brief(res.get("related_images")),
        "n_images": len(res.get("related_images") or []),
        "has_math_lesson": bool(res.get("math_lesson")),
        "has_science_experiment": bool(res.get("science_experiment")),
        "ms": res.get("ms"),
        "error": res.get("error"),
        "path": res.get("path"),
        "scores": score_row(q, res, channel),
    }


def run_channel(session: AuthSession, bank: dict, channel: str) -> dict:
    runner = chat_tutor if channel == "tutor" else chat_voice
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
        primary = pack_primary(q, res, channel)
        primaries.append(primary)
        print(f"  [{channel}] {q['id']} -> {primary['scores']['overall']}", flush=True)

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
                "answer": fu_res.get("answer") or "",
                "answer_clip": clip(fu_res.get("answer"), 450),
                "images": img_brief(fu_res.get("related_images")),
                "n_images": len(fu_res.get("related_images") or []),
                "has_math_lesson": bool(fu_res.get("math_lesson")),
                "has_science_experiment": bool(fu_res.get("science_experiment")),
                "ms": fu_res.get("ms"),
                "error": fu_res.get("error"),
                "path": fu_res.get("path"),
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
    ap.add_argument("--channel", choices=("tutor", "voice", "both"), default="both")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    bank = load_bank(args.chapter)
    print(f"=== {bank['slug']} / {bank['chapter']} ===", flush=True)
    require_api()
    session = AuthSession()
    nest = probe_nest(session, bank)
    print("nest_probe", nest.get("ok"), nest.get("error"), flush=True)

    empty = {"primaries": [], "followups": []}
    tutor = empty
    voice = empty
    if args.channel in ("tutor", "both"):
        print("AI TUTOR", flush=True)
        tutor = run_channel(session, bank, "tutor")
    if args.channel in ("voice", "both"):
        print("AI VOICE", flush=True)
        voice = run_channel(session, bank, "voice")

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
        },
        "nest_probe": nest,
        "tutor": tutor,
        "voice": voice,
        "tutor_avg": avg_channel(tutor) if tutor["primaries"] else 0.0,
        "voice_avg": avg_channel(voice) if voice["primaries"] else 0.0,
    }

    md_path = RESULTS / f"{bank['slug']}.md"
    write_chapter_md(md_path, payload)
    raw_path = RAW / f"{bank['slug']}.json"
    raw_path.write_text(json.dumps(slim_for_raw(payload), indent=2))
    print("Wrote", md_path)
    print("Wrote", raw_path)
    print("TUTOR", payload["tutor_avg"], "VOICE", payload["voice_avg"])


if __name__ == "__main__":
    main()

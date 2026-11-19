#!/usr/bin/env python3
"""Replay a multi-turn transcript bank against AI Voice / AI Tutor and score every turn with an LLM judge.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/replay_transcript.py                       # voice, default bank
  .venv/bin/python suneel_chapter_qa/replay_transcript.py --channel tutor
  .venv/bin/python suneel_chapter_qa/replay_transcript.py --only R07,R21 --tag after_fixes

Turns run in order with the real conversation history, and the tutor channel carries the
per-chat state the /ai-tutor page sends back (quiz_state, prefers_short, explained_points).
Each turn gets deterministic checks (forbidden patterns, sentence/word caps) plus a 0-10 judge
score against its "expect" line. Exit code 1 when any turn scores below --pass-score.
Needs the API running (QA_API, default http://127.0.0.1:8000) and the backend .env for the judge.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from qa_lib.api import AuthSession, chat_tutor, chat_voice, clip, require_api  # noqa: E402

BANKS = ROOT / "banks"
DEFAULT_BANK = "social_ch2_reshaping_regression"
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

JUDGE_PROMPT = """You are grading one turn of an AI tutor talking to a Class 8 student{mode_note}.
Chapter: {chapter}
What the chapter covers: {summary}

RECENT CONVERSATION:
{history}

STUDENT: {query}
TUTOR: {answer}

EXPECTED BEHAVIOUR: {expect}

Score 0-10: 10 = does exactly what is expected, accurate, kind, the right length for a student;
7 = acceptable with small issues; 4 = misses the main expectation; 0 = wrong, harmful or off-task.
Reply with one line only: <score> | <one short reason>"""


def deterministic_issues(turn: dict, answer: str, channel: str) -> list[str]:
    issues = []
    for pat in turn.get("forbid", []):
        if re.search(pat, answer):
            issues.append(f"forbidden pattern {pat!r}")
    if turn.get("max_sentences"):
        n = len(_SENTENCE_END.findall(answer.strip()))
        if n > turn["max_sentences"]:
            issues.append(f"{n} sentences > {turn['max_sentences']}")
    if turn.get("max_words") and len(answer.split()) > turn["max_words"]:
        issues.append(f"{len(answer.split())} words > {turn['max_words']}")
    if channel == "voice" and re.search(r"\*\*(Topic|Key Points|Remember|Try This)\*\*", answer):
        issues.append("written card in a voice reply")
    if re.search(r"(?i)context (?:isn'?t|is not) available|the tutor (?:said|mentioned|used)", answer):
        issues.append("broke character (context unavailable / 'the tutor')")
    return issues


async def judge(turn: dict, answer: str, history: list[dict], bank: dict, channel: str) -> tuple[int, str]:
    from app.services import llm_client

    hist = "\n".join(f"{t['role'].upper()}: {clip(t['content'], 300)}" for t in history[-6:]) or "(start)"
    prompt = JUDGE_PROMPT.format(
        mode_note=" by voice (replies are spoken aloud)" if channel == "voice" else " in text chat",
        chapter=bank["chapter"],
        summary=bank.get("chapter_summary") or "(see the chapter title)",
        history=hist,
        query=turn["query"],
        answer=answer,
        expect=turn["expect"],
    )
    last = "unparseable reply"
    for _ in range(2):
        try:
            raw = await llm_client.complete([{"role": "user", "content": prompt}], feature="chat", max_tokens=60)
            m = re.match(r"\s*(\d{1,2})\s*\|\s*(.*)", raw or "")
            if m:
                return min(10, int(m.group(1))), m.group(2).strip()
        except Exception as exc:
            last = str(exc)
    return -1, f"judge failed: {last}"


def run(args: argparse.Namespace) -> int:
    bank = json.loads((BANKS / f"{args.bank}.json").read_text())
    only = {x.strip() for x in args.only.split(",")} if args.only else None
    require_api()
    session = AuthSession()
    call = chat_voice if args.channel == "voice" else chat_tutor
    history: list[dict] = []
    state: dict = {}
    rows = []
    for turn in bank["turns"]:
        extra = dict(state) if args.channel == "tutor" else {}
        if args.channel == "voice" and state.get("prefers_short"):
            extra["prefers_short"] = True
        res = call(
            session,
            query=turn["query"],
            history=history[-8:],
            board=bank["board"],
            class_level=bank["class_level"],
            subject=bank["subject"],
            chapter_id=bank["chapter_id"],
            chapter=bank["chapter"],
            extra=extra,
        )
        answer = (res.get("answer") or "").strip()
        meta = res.get("meta") or {}
        if args.channel == "tutor":
            state = {
                "quiz_state": meta.get("quiz_state"),
                "prefers_short": meta.get("prefers_short", state.get("prefers_short", False)),
                "explained_points": meta.get("explained_points", state.get("explained_points")),
            }
        elif re.search(r"(?i)stop giving long|keep it short|too long", turn["query"]):
            state["prefers_short"] = True
        nest_owned = args.channel == "voice" and turn.get("owner") == "nest"
        if (only is None or turn["id"] in only) and not nest_owned:
            issues = deterministic_issues(turn, answer, args.channel)
            score, reason = asyncio.run(judge(turn, answer, history, bank, args.channel))
            if issues and score > 4:
                score = 4
            rows.append({
                "id": turn["id"], "query": turn["query"], "answer": answer, "score": score,
                "reason": reason, "issues": issues, "ms": res.get("ms"), "error": res.get("error"),
            })
            flag = "PASS" if score >= args.pass_score else "FAIL"
            print(f"[{flag}] {turn['id']} {score:>2}/10  {clip(turn['query'], 60)}")
            if flag == "FAIL":
                print(f"        tutor: {clip(answer, 220)}")
                print(f"        why:   {reason}{' | ' + '; '.join(issues) if issues else ''}")
        history += [{"role": "user", "content": turn["query"]}, {"role": "assistant", "content": answer}]
        time.sleep(args.pause)

    scored = [r["score"] for r in rows if r["score"] >= 0]
    avg = sum(scored) / len(scored) if scored else 0.0
    failing = [r["id"] for r in rows if r["score"] < args.pass_score]
    print(f"\n{args.channel}: average {avg:.1f}/10 over {len(rows)} turns; failing: {', '.join(failing) or 'none'}")

    out_dir = ROOT / "runs" / (args.tag or "replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bank}_{args.channel}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"bank": args.bank, "channel": args.channel, "average": avg, "turns": rows}, indent=2))
    print(f"saved {out}")
    return 1 if failing else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--channel", choices=("voice", "tutor"), default="voice")
    ap.add_argument("--only", default="", help="comma-separated turn ids to score (all turns still run)")
    ap.add_argument("--pass-score", type=int, default=8)
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between turns (LLM rate limits)")
    ap.add_argument("--tag", default="")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Struggling-learner stress test: text vs voice on Light chapter.

Persona: lazy Class 10 student, weak foundations, easily confused, short attention.
Runs the same multi-turn script on both channels, scores for learner clarity,
and writes a markdown + JSON report.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/run_struggling_learner_eval.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from qa_lib.api import AuthSession, chat_tutor, chat_voice, clip, require_api  # noqa: E402

OUT_MD = ROOT / "results" / "struggling_learner_eval.md"
OUT_JSON = ROOT / "raw" / "struggling_learner_eval.json"

BANK = {
    "board": "CBSE",
    "class_level": "CLASS_10",
    "subject": "Science",
    "chapter_id": "647",
    "chapter": "9. Light – Reflection and Refraction",
}

# Multi-turn script as a struggling / lazy learner (shared history per channel).
TURNS: list[dict] = [
    {
        "id": "T1_topics",
        "query": "idk whats in this chapter tell me topics",
        "need": "topic inventory / orientation",
        "expect": ["reflection", "refraction", "mirror", "lens"],
        "avoid": ["quick quiz", "science-experiment"],
    },
    {
        "id": "T2_confused_def",
        "query": "wait what is refraction i keep forgetting",
        "need": "simple definition + everyday hook",
        "expect": ["bend", "light", "medium"],
        "avoid": ["according to the chapter", "in conclusion"],
    },
    {
        "id": "T3_lazy_yes",
        "query": "ok",
        "need": "gentle next step, not a quiz dump if offer was specific",
        "expect": [],
        "avoid": [],
    },
    {
        "id": "T4_why",
        "query": "why it bends tho??",
        "need": "cause in plain words, no jargon pile",
        "expect": ["speed", "light", "medium"],
        "avoid": ["sin", "snell's"],
    },
    {
        "id": "T5_glass_slab",
        "query": "glass slab thing is confusing explain like im dumb",
        "need": "scaffolded glass-slab idea",
        "expect": ["glass", "bend", "parallel"],
        "avoid": ["activity 9.10"],
    },
    {
        "id": "T6_check",
        "query": "so emergent ray same direction? yes or no",
        "need": "direct yes/no then one-line why",
        "expect": ["yes", "parallel"],
        "avoid": [],
    },
    {
        "id": "T7_give_up",
        "query": "still confused about normal line what is that",
        "need": "re-explain 'normal' simply",
        "expect": ["straight", "surface", "normal"],
        "avoid": [],
    },
    {
        "id": "T8_mirror_mixup",
        "query": "concave and convex same na?",
        "need": "correct misconception gently",
        "expect": ["concave", "convex", "different"],
        "avoid": [],
    },
]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _avg_sentence_len(text: str) -> float:
    parts = [p.strip() for p in re.split(r"[.!?]+", text or "") if p.strip()]
    if not parts:
        return 0.0
    return statistics.mean(len(_words(p)) for p in parts)


def _jargon_hits(text: str) -> list[str]:
    hard = [
        "according to the chapter",
        "in everyday terms",
        "snell's law",
        "sin i",
        "sin r",
        "lateral displacement",
        "optical density",
        "magnification",
        "cartesian sign",
    ]
    a = (text or "").lower()
    return [j for j in hard if j in a]


def score_for_struggling_learner(
    *,
    answer: str,
    channel: str,
    turn: dict,
    images: list | None,
) -> dict:
    """1–10 scores from a weak learner's point of view."""
    a = answer or ""
    al = a.lower()
    words = _words(a)
    n = len(words)
    bugs: list[str] = []

    # Clarity: short sentences, plain words
    sent_len = _avg_sentence_len(a)
    clarity = 8.0
    if sent_len > 18:
        clarity -= 2.0
        bugs.append(f"long sentences (~{sent_len:.0f} words)")
    if sent_len > 24:
        clarity -= 2.0
    jargon = _jargon_hits(a)
    if jargon:
        clarity -= min(3.0, 1.0 * len(jargon))
        bugs.append("hard jargon: " + ", ".join(jargon[:3]))
    if n < 12:
        clarity -= 3.0
        bugs.append("too thin to learn from")
    clarity = max(1.0, min(10.0, clarity))

    # Scaffolding: explains, checks, doesn't dump
    scaffold = 7.0
    if any(x in al for x in ("for example", "like when", "think of", "imagine", "like a")):
        scaffold += 1.0
    if "?" in a:
        scaffold += 0.5
    if channel == "text" and a.count("\n-") + a.count("\n*") >= 2:
        scaffold += 0.5
    if n > 280 and channel == "voice":
        scaffold -= 1.5
        bugs.append("voice turn too long to hold attention")
    if n > 450 and channel == "text" and turn["id"] != "T1_topics":
        scaffold -= 1.0
        bugs.append("text dump may overwhelm")
    if "```science-experiment" in al:
        scaffold -= 2.0
        bugs.append("science-experiment JSON dumped into answer")
    scaffold = max(1.0, min(10.0, scaffold))

    # Concept hit: expected idea words present
    expect = turn.get("expect") or []
    hits = sum(1 for k in expect if k.lower() in al) if expect else 0
    if expect:
        concept = round(10.0 * hits / max(1, len(expect)), 1)
        if hits == 0:
            bugs.append("missed key concept words: " + ", ".join(expect))
            concept = max(2.0, concept)
    else:
        concept = 7.0 if n >= 20 else 4.0

    for bad in turn.get("avoid") or []:
        b = (bad or "").lower().strip()
        if not b:
            continue
        # Short tokens ("sin") must be whole words — else they hit "using"/"lens".
        hit = bool(re.search(rf"\b{re.escape(b)}\b", al)) if len(b) <= 4 else (b in al)
        if hit:
            concept = max(1.0, concept - 2.0)
            bugs.append(f"unhelpful content: {bad}")

    # Empathy / tone for confused student
    empathy = 7.0
    if any(x in al for x in ("dumb", "obvious", "simply put you should", "as you know")):
        empathy -= 3.0
        bugs.append("tone may shame learner")
    if any(x in al for x in ("it's okay", "no worries", "lets break", "let's break", "step by step", "slowly")):
        empathy += 1.5
    if turn["id"] in ("T2_confused_def", "T5_glass_slab", "T7_give_up") and empathy < 8:
        if "confus" in al or "okay" in al or "alright" in al:
            empathy += 0.5
    empathy = max(1.0, min(10.0, empathy))

    # Usability of format for this channel
    fmt = 8.0
    if channel == "voice":
        if re.search(r"^\s*[-*]\s", a, re.M) or "**" in a:
            fmt -= 2.0
            bugs.append("markdown/bullets in voice")
        if n > 220:
            fmt -= 1.0
    else:
        if turn["id"] == "T1_topics" and not re.search(r"\d+\.", a):
            fmt -= 2.0
            bugs.append("topic list not clearly numbered")
        if images:
            fmt += 0.5
    fmt = max(1.0, min(10.0, fmt))

    overall = round(0.30 * clarity + 0.25 * scaffold + 0.25 * concept + 0.10 * empathy + 0.10 * fmt, 1)
    can_learn = overall >= 7.0 and concept >= 6.0 and clarity >= 6.0

    return {
        "clarity": round(clarity, 1),
        "scaffolding": round(scaffold, 1),
        "concept": round(concept, 1),
        "empathy": round(empathy, 1),
        "format": round(fmt, 1),
        "overall": overall,
        "can_weak_learner_get_it": can_learn,
        "words": n,
        "avg_sentence_words": round(sent_len, 1),
        "images": len(images or []),
        "bugs": bugs,
    }


def run_channel(session: AuthSession, channel: str) -> list[dict]:
    history: list[dict] = []
    rows: list[dict] = []
    chat = chat_voice if channel == "voice" else chat_tutor
    for turn in TURNS:
        t0 = time.time()
        res = chat(
            session,
            query=turn["query"],
            history=history,
            board=BANK["board"],
            class_level=BANK["class_level"],
            subject=BANK["subject"],
            chapter_id=BANK["chapter_id"],
            chapter=BANK["chapter"],
        )
        answer = (res.get("answer") or "").strip()
        images = res.get("related_images") or []
        scores = score_for_struggling_learner(
            answer=answer, channel=channel, turn=turn, images=images
        )
        row = {
            "id": turn["id"],
            "channel": channel,
            "query": turn["query"],
            "need": turn["need"],
            "answer": answer,
            "answer_clip": clip(answer, 500),
            "ms": res.get("ms") or int((time.time() - t0) * 1000),
            "ttft_ms": res.get("ttft_ms"),
            "error": res.get("error"),
            "scores": scores,
            "images": [
                {
                    "figure": im.get("figure_number"),
                    "activity": im.get("activity_number"),
                    "caption": clip(im.get("caption") or "", 80),
                }
                for im in images[:3]
            ],
        }
        rows.append(row)
        history.append({"role": "user", "content": turn["query"]})
        history.append({"role": "assistant", "content": answer[:2000]})
        print(
            f"[{channel}] {turn['id']} overall={scores['overall']} "
            f"learn={scores['can_weak_learner_get_it']} words={scores['words']} "
            f"bugs={scores['bugs'][:2]}"
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ["clarity", "scaffolding", "concept", "empathy", "format", "overall"]
    avg = {k: round(statistics.mean(r["scores"][k] for r in rows), 1) for k in keys}
    learn_rate = round(
        100.0 * sum(1 for r in rows if r["scores"]["can_weak_learner_get_it"]) / len(rows), 1
    )
    all_bugs: list[str] = []
    for r in rows:
        for b in r["scores"]["bugs"]:
            all_bugs.append(f"{r['id']}: {b}")
    return {
        "avg": avg,
        "learn_rate_pct": learn_rate,
        "turns": len(rows),
        "bugs": all_bugs,
    }


def write_report(payload: dict) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Struggling-learner eval — voice vs text",
        "",
        f"**Chapter:** {BANK['chapter']} (id {BANK['chapter_id']})",
        f"**Persona:** lazy Class 10 student, weak foundations, easily confused",
        f"**Turns:** {len(TURNS)} shared script per channel",
        "",
        "## Scorecard (avg /10)",
        "",
        "| Channel | Clarity | Scaffold | Concept | Empathy | Format | Overall | Learner-can-get-it |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ch in ("text", "voice"):
        s = payload["summary"][ch]
        a = s["avg"]
        lines.append(
            f"| {ch} | {a['clarity']} | {a['scaffolding']} | {a['concept']} | "
            f"{a['empathy']} | {a['format']} | **{a['overall']}** | {s['learn_rate_pct']}% |"
        )

    lines += ["", "## Verdict", "", payload["verdict"], "", "## Gaps to improve", ""]
    for g in payload["gaps"]:
        lines.append(f"- {g}")

    lines += ["", "## Turn detail", ""]
    for ch in ("text", "voice"):
        lines.append(f"### {ch.upper()}")
        for r in payload["channels"][ch]:
            sc = r["scores"]
            lines.append(
                f"- **{r['id']}** — overall **{sc['overall']}** "
                f"(clarity {sc['clarity']}, concept {sc['concept']}, "
                f"learn={sc['can_weak_learner_get_it']})"
            )
            lines.append(f"  - Student: _{r['query']}_")
            lines.append(f"  - Tutor: {r['answer_clip']}")
            if sc["bugs"]:
                lines.append(f"  - Gaps: {'; '.join(sc['bugs'])}")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


def build_gaps(payload: dict) -> list[str]:
    gaps: list[str] = []
    for ch in ("text", "voice"):
        s = payload["summary"][ch]
        if s["avg"]["clarity"] < 7.5:
            gaps.append(f"{ch}: clarity under 7.5 — shorten sentences / cut jargon for weak learners")
        if s["avg"]["scaffolding"] < 7.5:
            gaps.append(f"{ch}: scaffolding weak — add analogy + one check question more often")
        if s["learn_rate_pct"] < 75:
            gaps.append(f"{ch}: only {s['learn_rate_pct']}% of turns were learnable for a weak student")
        for b in s["bugs"]:
            if "jargon" in b or "science-experiment" in b or "markdown" in b or "thin" in b:
                gaps.append(f"{ch}: {b}")
    # Cross-channel
    tv = payload["summary"]["text"]["avg"]["overall"]
    vv = payload["summary"]["voice"]["avg"]["overall"]
    if abs(tv - vv) >= 1.5:
        better = "text" if tv > vv else "voice"
        gaps.append(
            f"Modes diverge by {abs(tv-vv):.1f} points — prefer {better} for deep clarity, "
            "use the other for engagement / first pass"
        )
    # Dedup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for g in gaps:
        if g not in seen:
            seen.add(g)
            out.append(g)
    if not out:
        out.append("No critical gaps flagged by heuristics — still spot-check misconception turns manually.")
    return out[:12]


def main() -> None:
    require_api()
    session = AuthSession()
    print(f"Logged in as {session.login_as}")
    print(f"Chapter {BANK['chapter']} ({BANK['chapter_id']})")

    channels: dict[str, list[dict]] = {}
    for ch in ("text", "voice"):
        print(f"\n=== {ch.upper()} ===")
        channels[ch] = run_channel(session, ch)

    summary = {ch: summarize(rows) for ch, rows in channels.items()}
    payload = {
        "bank": BANK,
        "persona": "struggling / lazy Class 10 learner with weak foundations",
        "channels": channels,
        "summary": summary,
        "gaps": [],
        "verdict": "",
    }
    payload["gaps"] = build_gaps(payload)

    t = summary["text"]["avg"]["overall"]
    v = summary["voice"]["avg"]["overall"]
    tl = summary["text"]["learn_rate_pct"]
    vl = summary["voice"]["learn_rate_pct"]
    if t >= 7.5 and v >= 7.0 and tl >= 70 and vl >= 60:
        verdict = (
            f"Mostly satisfies a struggling learner for this chapter "
            f"(text {t}/10, voice {v}/10). Text is safer for clarity; voice is usable for first contact."
        )
    elif t >= 6.5 or v >= 6.5:
        verdict = (
            f"Partial: student can get pieces but not reliable clarity "
            f"(text {t}/10 @ {tl}% learnable, voice {v}/10 @ {vl}% learnable). See gaps."
        )
    else:
        verdict = (
            f"Not yet satisfying for struggling learners "
            f"(text {t}/10, voice {v}/10). Foundational scaffolding needs work."
        )
    payload["verdict"] = verdict
    write_report(payload)
    print("\n" + verdict)
    print("Gaps:")
    for g in payload["gaps"]:
        print(" -", g)


if __name__ == "__main__":
    main()

"""Probe: explain a chapter, then ask twice "do you miss anything?"

Shows two things the product currently hides:

  1. How each turn is routed before the model speaks (intent, retrieval query).
  2. Whether the self-check actually fills gaps in the chapter's own topic list,
     or just repeats, denies, or wanders.

Calls the same ``chapter_aware_qa`` path as ``POST /auth/chat``, in-process,
so it does not open a second student login (single-session would reject it).

  cd ai-tutor-backend
  .venv/bin/python scripts/miss_anything_probe.py
  .venv/bin/python scripts/miss_anything_probe.py --chapter-id 37

Writes evaluation/reports/miss_anything_probe.{md,json}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.conversation_context import resolve_conversation_context  # noqa: E402
from app.services.voice_chapter_vocab import is_chapter_overview_request  # noqa: E402

OUT_DIR = ROOT / "evaluation" / "reports"
OUT_JSON = OUT_DIR / "miss_anything_probe.json"
OUT_MD = OUT_DIR / "miss_anything_probe.md"

EXPLAIN = "explain this chapter"
MISS = "do you miss anything?"

_STOP = frozenset(
    "the a an of and or to in on for with from by at as is are was were be "
    "this that these those it its into about your you we our their".split()
)
_DENIAL = re.compile(
    r"\b("
    r"didn'?t miss|did not miss|don'?t miss|do not miss|nothing (?:else|missing)|"
    r"that(?:'s| is) (?:everything|all)|covered (?:everything|it all|all of)|"
    r"no[,.]?\s+i (?:didn'?t|did not|don'?t)|i (?:included|covered) (?:all|every)|"
    r"nothing was left|that covers"
    r")\b",
    re.I,
)
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) >= 4 and w not in _STOP}


def _topic_hit(title: str, answer: str) -> bool:
    title_l = (title or "").lower().strip()
    answer_l = (answer or "").lower()
    if len(title_l) >= 6 and title_l in answer_l:
        return True
    tokens = [w for w in _WORD.findall(title_l) if len(w) >= 4 and w not in _STOP]
    if not tokens:
        return title_l in answer_l if title_l else False
    hits = sum(1 for w in tokens if w in answer_l)
    need = 1 if len(tokens) == 1 else max(2, int(round(len(tokens) * 0.6)))
    return hits >= need


def _route(query: str, history: list[dict], chapter: str) -> dict:
    ctx = resolve_conversation_context(query, conversation_history=history, chapter=chapter)
    return {
        "intent": ctx.followup_type,
        "method": ctx.intent_method,
        "confidence": ctx.intent_confidence,
        "response_mode": ctx.response_mode.value,
        "retrieval_query": ctx.retrieval_query,
        "resolved_topic": ctx.resolved_topic,
        "overview_prompt": is_chapter_overview_request(query),
    }


def _coverage(topics: list[dict], answer: str) -> dict:
    rows = []
    for t in topics:
        title = t.get("title") or ""
        hit = _topic_hit(title, answer)
        rows.append({
            "title": title,
            "section": t.get("section_number") or "",
            "hit": hit,
        })
    missed = [r for r in rows if not r["hit"]]
    return {
        "n_topics": len(rows),
        "n_hit": sum(1 for r in rows if r["hit"]),
        "missed": [r["title"] for r in missed],
        "rows": rows,
    }


def _new_vs(prev: str, cur: str) -> dict:
    pw, cw = _words(prev), _words(cur)
    if not cw:
        return {"new_word_ratio": 0.0, "overlap_ratio": 0.0, "n_new_words": 0}
    new = cw - pw
    return {
        "new_word_ratio": round(len(new) / len(cw), 2),
        "overlap_ratio": round(len(cw & pw) / len(cw), 2),
        "n_new_words": len(new),
    }


def _pick_chapter(chapter_id: str | None) -> dict:
    from sqlalchemy import text

    from app.core.database import SessionLocal

    with SessionLocal() as db:
        if chapter_id:
            rows = db.execute(
                text(
                    "SELECT id, board::text, class_level::text, subject_name, chapter "
                    "FROM textbook_uploads WHERE id = :id"
                ),
                {"id": int(chapter_id)},
            ).fetchall()
        else:
            rows = db.execute(
                text(
                    "SELECT id, board::text, class_level::text, subject_name, chapter "
                    "FROM textbook_uploads "
                    "WHERE class_level::text = 'CLASS_8' AND subject_name ILIKE '%science%' "
                    "ORDER BY id"
                )
            ).fetchall()
    if not rows:
        raise SystemExit("no textbook chapter found")
    # Prefer a lesson that actually has section headings. Chapter 1 of Curiosity
    # is a short intro and yields an empty topic list, so the gap check is blind.
    preferred = ("health", "why do we fall ill", "microorganism", "electricity")

    def _row(candidate) -> dict:
        return {
            "board": candidate[1] or "CBSE",
            "class_level": candidate[2] or "CLASS_8",
            "subject": candidate[3] or "",
            "chapter_id": str(candidate[0]),
            "chapter": (candidate[4] or "").replace("\x08", "").strip(),
        }

    if chapter_id:
        return _row(rows[0])
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if any(p in (r[4] or "").lower() for p in preferred) else 1,
            -(len(r[4] or "")),
        ),
    )
    for candidate in ranked:
        ch = _row(candidate)
        try:
            n = len(_load_topics(ch))
        except Exception:
            n = 0
        print(f"  candidate {ch['chapter_id']} topics={n} {ch['chapter']}", flush=True)
        if n >= 4:
            return ch
    return _row(ranked[0])


def _load_topics(ch: dict) -> list[dict]:
    from app.modules.student_learning.topic_progress import list_chapter_topics

    return list_chapter_topics(
        chapter_id=int(ch["chapter_id"]),
        board=ch["board"],
        class_level=ch["class_level"],
        subject_name=ch["subject"],
        chapter_name=ch["chapter"],
    )


async def _ask(ch: dict, query: str, history: list[dict]) -> dict:
    from app.services.chapter_collection import resolve_chat_collection
    from app.services.chat_service.qa import chapter_aware_qa

    collection = resolve_chat_collection(
        ch["board"], ch["class_level"], ch["subject"], [ch["chapter_id"]]
    )
    t0 = time.time()
    answer, images, _math, _science = await chapter_aware_qa(
        query,
        collection_name=collection or "",
        chapter_ids=[ch["chapter_id"]],
        class_level=ch["class_level"],
        board=ch["board"],
        subject_name=ch["subject"],
        chapter=ch["chapter"],
        chapter_names=[ch["chapter"]],
        conversation_history=history,
        student_name="Probe",
        student_key="miss-anything-probe",
    )
    return {
        "query": query,
        "answer": answer or "",
        "n_images": len(images or []),
        "collection": collection,
        "ms": int((time.time() - t0) * 1000),
    }


def _judge_check(label: str, answer: str, missed_before: list[str], prev_answer: str) -> dict:
    filled = [t for t in missed_before if _topic_hit(t, answer)]
    still = [t for t in missed_before if t not in filled]
    return {
        "label": label,
        "chars": len(answer),
        "denies_gap": bool(_DENIAL.search(answer)),
        "gaps_before": missed_before,
        "gaps_filled": filled,
        "gaps_still_open": still,
        "vs_previous": _new_vs(prev_answer, answer),
    }


def _md(payload: dict) -> str:
    ch = payload["chapter"]
    lines = [
        "# Miss-anything probe",
        "",
        f"- **Chapter:** `{ch['chapter_id']}` {ch['chapter']} ({ch['subject']})",
        f"- **Topics in the book:** {payload['n_topics']}",
        "",
        "## How each turn was routed",
        "",
        "This is decided before the model writes a word. The retrieval query is what the textbook search actually runs.",
        "",
        "| Turn | Student said | Intent | Retrieval query | Overview prompt |",
        "|------|----------------|--------|-----------------|-----------------|",
    ]
    for turn in payload["turns"]:
        r = turn["route"]
        rq = (r["retrieval_query"] or "").replace("|", "/")
        lines.append(
            f"| {turn['label']} | {turn['query']} | `{r['intent']}` ({r['method']}) | {rq} | {r['overview_prompt']} |"
        )
    cov = payload["explain_coverage"]
    lines += [
        "",
        "## Turn 1 — explain this chapter",
        "",
        f"Covered **{cov['n_hit']} / {cov['n_topics']}** book topics.",
        "",
    ]
    if cov["missed"]:
        lines.append("Missed:")
        lines.append("")
        for t in cov["missed"]:
            lines.append(f"- {t}")
        lines.append("")
    lines.append("### Answer")
    lines.append("")
    lines.append(payload["turns"][0]["answer"].strip() or "_(empty)_")
    lines.append("")
    for turn, check in zip(payload["turns"][1:], payload["checks"]):
        lines += [
            f"## {turn['label']} — {turn['query']}",
            "",
            f"- Denied a gap: **{check['denies_gap']}**",
            f"- Gaps filled this turn: {', '.join(check['gaps_filled']) or 'none'}",
            f"- Gaps still open: {', '.join(check['gaps_still_open']) or 'none'}",
            f"- New words vs previous answer: {check['vs_previous']['new_word_ratio']} "
            f"({check['vs_previous']['n_new_words']} words; overlap {check['vs_previous']['overlap_ratio']})",
            f"- {turn['chars']} characters, {turn['ms']} ms",
            "",
            "### Answer",
            "",
            turn["answer"].strip() or "_(empty)_",
            "",
        ]
    lines += ["## What this run shows", ""]
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _notes(payload: dict) -> list[str]:
    notes: list[str] = []
    explain = payload["turns"][0]
    checks = payload["checks"]
    cov = payload["explain_coverage"]
    if cov["n_topics"] and cov["n_hit"] < cov["n_topics"]:
        notes.append(
            f"The first explanation covered {cov['n_hit']} of {cov['n_topics']} book topics, "
            "so a real gap existed before the student asked."
        )
    elif cov["n_topics"]:
        notes.append("The first explanation already named every book topic the checker could see.")
    else:
        notes.append("No book topic list loaded, so gap checks are only repetition and denial.")

    for turn in payload["turns"][1:]:
        intent = turn["route"]["intent"]
        rq = turn["route"]["retrieval_query"]
        if intent == "new_topic":
            notes.append(
                f"'{turn['query']}' was routed as a new topic. "
                f"Textbook search ran on {rq!r} instead of the chapter the student just heard."
            )
        elif turn["route"].get("method") == "gap_check":
            notes.append(
                f"'{turn['query']}' was a gap check against the chapter topic list "
                f"(retrieval {rq!r})."
            )
        elif intent == "continue_explanation" and turn["route"]["overview_prompt"] is False:
            notes.append(
                f"'{turn['query']}' continued the previous explanation, "
                "but no prompt told the model to compare its last answer with the chapter topic list."
            )
    for check in checks:
        if check["denies_gap"] and check["gaps_still_open"]:
            notes.append(
                f"{check['label']} said nothing was missing while these topics were still absent: "
                + ", ".join(check["gaps_still_open"])
                + "."
            )
        elif check["gaps_filled"]:
            notes.append(
                f"{check['label']} did add previously missing topics: "
                + ", ".join(check["gaps_filled"])
                + "."
            )
        elif not check["denies_gap"] and check["gaps_still_open"]:
            notes.append(
                f"{check['label']} wrote more text but did not name the still-missing topics "
                f"({', '.join(check['gaps_still_open'])}). "
                f"New-word ratio vs the previous answer: {check['vs_previous']['new_word_ratio']}."
            )
        if check["vs_previous"]["overlap_ratio"] >= 0.75 and check["vs_previous"]["n_new_words"] < 12:
            notes.append(f"{check['label']} mostly repeated the previous answer.")
    if explain["route"]["overview_prompt"] and cov["n_topics"] and cov["n_hit"] < max(3, cov["n_topics"] // 2):
        notes.append(
            "Turn 1 matched the whole-chapter prompt (walk every topic) but still skipped most of the list. "
            "The model is not being forced to tick the CHAPTER TOPICS block."
        )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter-id", default="")
    args = ap.parse_args()

    ch = _pick_chapter(args.chapter_id or None)
    if not ch:
        raise SystemExit("no chapters in catalog")
    print(f"chapter: {ch['chapter_id']} {ch['chapter']} ({ch['subject']})", flush=True)

    try:
        topics = _load_topics(ch)
    except Exception as exc:
        print(f"topic list failed: {exc}", flush=True)
        topics = []
    print(f"book topics: {len(topics)}", flush=True)

    history: list[dict] = []
    turns = []
    plan = [
        ("Turn 1 explain", EXPLAIN),
        ("Turn 2 check", MISS),
        ("Turn 3 check again", MISS),
    ]

    async def _run_turns() -> None:
        for label, query in plan:
            route = _route(query, history, ch["chapter"])
            print(f"{label}: intent={route['intent']} retrieval={route['retrieval_query']!r}", flush=True)
            got = await _ask(ch, query, history)
            print(f"  {got['ms']} ms, {len(got['answer'])} chars", flush=True)
            turns.append({
                "label": label,
                "query": query,
                "answer": got["answer"],
                "chars": len(got["answer"]),
                "ms": got["ms"],
                "n_images": got["n_images"],
                "route": route,
                "collection": got.get("collection"),
            })
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": got["answer"]})

    asyncio.run(_run_turns())

    cov = _coverage(topics, turns[0]["answer"])
    missed = list(cov["missed"])
    checks = []
    prev = turns[0]["answer"]
    for turn in turns[1:]:
        check = _judge_check(turn["label"], turn["answer"], missed, prev)
        checks.append(check)
        missed = check["gaps_still_open"]
        prev = turn["answer"]

    payload = {
        "chapter": ch,
        "n_topics": len(topics),
        "topics": [{"title": t.get("title"), "section": t.get("section_number")} for t in topics],
        "turns": turns,
        "explain_coverage": cov,
        "checks": checks,
        "notes": [],
    }
    payload["notes"] = _notes(payload)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    OUT_MD.write_text(_md(payload))
    print(f"wrote {OUT_MD}", flush=True)
    for note in payload["notes"]:
        print(f"- {note}", flush=True)


if __name__ == "__main__":
    main()

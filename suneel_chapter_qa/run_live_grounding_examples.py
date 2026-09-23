#!/usr/bin/env python3
"""Live LLM smoke: student Q → tutor answer → interactive panel grounding.

  cd ai-tutor-backend && .venv/bin/python suneel_chapter_qa/run_live_grounding_examples.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from lib.api import AuthSession, chat_tutor, chat_voice, clip  # noqa: E402

OUT = ROOT / "results" / "live_grounding_examples.md"
RAW = ROOT / "raw" / "live_grounding_examples.json"

# Suneel CLASS_8 chapters (from banks)
CASES = [
    {
        "id": "M-open-box",
        "subject": "Mathematics",
        "chapter_id": "38",
        "chapter": "Chapter-1 : A SQUARE AND A CUBE",
        "board": "CBSE",
        "class_level": "CLASS_8",
        "query": (
            "A rectangular sheet of paper is 30 cm long and 20 cm wide. "
            "Four squares of side 5 cm are cut from its four corners. "
            "The remaining sheet is folded upwards to make an open box. "
            "Find the volume of the box with steps."
        ),
        "expect_nums": {5, 10, 20, 30, 1000},
        "expect_types": {"area-resizer", "mensuration-cube", "concept-explorer"},
        "expect_sliders": {"length": 20, "width": 10, "height": 5, "cut": 5},
        "kind": "math",
    },
    {
        "id": "M-powers",
        "subject": "Mathematics",
        "chapter_id": "39",
        "chapter": "Chapter-2 : POWER PLAY",
        "board": "CBSE",
        "class_level": "CLASS_8",
        "query": "ok explain 2 to the power 5 slowly with the expanded form and final value",
        "expect_nums": {2, 5, 32},
        "expect_types": {"concept-explorer", "algebra-stepper", "number-line"},
        "forbid_types": {"linear-graph", "statistics-lab"},
        "kind": "math",
    },
    {
        "id": "M-square-81",
        "subject": "Mathematics",
        "chapter_id": "38",
        "chapter": "Chapter-1 : A SQUARE AND A CUBE",
        "board": "CBSE",
        "class_level": "CLASS_8",
        "query": "explain with steps how to check if 81 is a perfect square",
        "expect_nums": {81, 9},
        "expect_types": {"factor-tree", "mensuration-cube", "concept-explorer", "place-value"},
        "kind": "math",
    },
    {
        "id": "S-pressure",
        "subject": "Science",
        "chapter_id": "35",
        "chapter": "Chapter 1 - Exploring the Investigative World of Science",
        "board": "CBSE",
        "class_level": "CLASS_8",
        "query": (
            "A rectangular wooden block weighs 40 N. It is placed on a table in two positions. "
            "Position A: area 200 cm². Position B: area 100 cm². "
            "Calculate the pressure in each position and say which is greater. Explain with formula P=F/A."
        ),
        "expect_nums": {40, 200, 100, 0.2, 0.4},
        "expect_types": {"force-pressure-lab"},
        "kind": "science",
    },
    {
        "id": "S-disease",
        "subject": "Science",
        "chapter_id": "37",
        "chapter": "Chapter 3 - Health: The Ultimate Treasure",
        "board": "CBSE",
        "class_level": "CLASS_8",
        "query": "how does disease spread? explain like steps and how washing hands helps",
        "expect_nums": set(),
        "expect_types": {"disease-transmission-simulator", "concept-explorer", "human-body-system-3d"},
        "forbid_types": {"circuit-builder", "linear-graph"},
        "kind": "science",
    },
    {
        "id": "S-photosynthesis",
        "subject": "Science",
        "chapter_id": "35",
        "chapter": "Chapter 1 - Exploring the Investigative World of Science",
        "board": "CBSE",
        "class_level": "CLASS_8",
        "query": (
            "Even if not the main chapter focus, explain photosynthesis as an experiment: "
            "what goes in, what comes out, and show an interactive lab if you can."
        ),
        "expect_nums": set(),
        "expect_types": {"plant-anatomy-lab", "concept-explorer"},
        "kind": "science",
    },
]


def panel_info(res: dict, kind: str) -> dict:
    if kind == "math":
        lesson = res.get("math_lesson")
        if not lesson:
            return {"has": False}
        viz = lesson.get("visualization") or {}
        return {
            "has": True,
            "type": viz.get("visualizationType"),
            "title": viz.get("title"),
            "sliders": [
                {"id": s.get("id"), "default": s.get("default"), "min": s.get("min"), "max": s.get("max")}
                for s in (viz.get("sliders") or [])
                if isinstance(s, dict)
            ],
            "calcs": [
                {"id": c.get("id"), "formula": c.get("formula")}
                for c in (viz.get("liveCalculations") or [])
                if isinstance(c, dict)
            ],
        }
    exp = res.get("science_experiment")
    if not exp:
        return {"has": False}
    e = exp.get("experiment") or exp
    return {
        "has": True,
        "type": e.get("experimentType"),
        "title": e.get("title"),
        "sliders": [
            {"id": s.get("id"), "default": s.get("default"), "min": s.get("min"), "max": s.get("max")}
            for s in (e.get("sliders") or [])
            if isinstance(s, dict)
        ],
        "calcs": [
            {"id": c.get("id"), "formula": c.get("formula")}
            for c in (e.get("liveCalculations") or e.get("calcs") or [])
            if isinstance(c, dict)
        ],
        "equation": ((e.get("threeViews") or {}).get("scientific") or {}).get("equation"),
    }


def score_case(case: dict, res: dict, info: dict) -> dict:
    bugs: list[str] = []
    answer = (res.get("answer") or "").lower()
    if res.get("error"):
        return {"ok": False, "grade": "FAIL", "bugs": [res["error"][:200]], "num_hits": 0}

    if not info.get("has"):
        bugs.append("no interactive panel")
        grade = "FAIL"
    else:
        vt = info.get("type") or ""
        if case.get("forbid_types") and vt in case["forbid_types"]:
            bugs.append(f"forbidden type {vt}")
        if case.get("expect_types") and vt not in case["expect_types"]:
            bugs.append(f"unexpected type {vt} (wanted {sorted(case['expect_types'])})")
        defaults = [float(s["default"]) for s in info.get("sliders") or [] if s.get("default") is not None]
        expect = case.get("expect_nums") or set()
        # allow float equality for 0.2 / 0.4
        def hit(n):
            for d in defaults:
                if abs(float(n) - d) < 1e-6:
                    return True
            # also accept numbers present in answer prose if panel exists
            if str(n) in answer or f"{float(n):.1f}" in answer:
                return True
            return False

        num_hits = sum(1 for n in expect if hit(n)) if expect else None
        if expect and defaults:
            # at least half of expect nums should appear in defaults OR answer
            if num_hits < max(1, len(expect) // 2):
                bugs.append(f"weak number grounding hits={num_hits}/{len(expect)} defaults={defaults}")
        if expect and info.get("has") and not defaults and case["kind"] == "math":
            bugs.append("panel has no sliders to carry numbers")

        want = case.get("expect_sliders") or {}
        if want and info.get("has"):
            by = {
                str(s.get("id")): float(s["default"])
                for s in (info.get("sliders") or [])
                if s.get("id") is not None and s.get("default") is not None
            }
            for sid, val in want.items():
                got = by.get(sid)
                if got is None or abs(got - float(val)) > 1e-6:
                    bugs.append(f"slider {sid}={got} want {val}")

        if bugs and any("forbidden" in b for b in bugs):
            grade = "FAIL"
        elif bugs and any("no interactive" in b for b in bugs):
            grade = "FAIL"
        elif bugs:
            grade = "PARTIAL"
        else:
            grade = "PASS"
        return {
            "ok": grade == "PASS",
            "grade": grade,
            "bugs": bugs,
            "num_hits": num_hits,
            "type": vt,
            "defaults": defaults,
        }

    return {"ok": False, "grade": grade, "bugs": bugs, "num_hits": 0}


def run_one(session: AuthSession, case: dict, channel: str) -> dict:
    runner = chat_tutor if channel == "tutor" else chat_voice
    t0 = time.time()
    res = runner(
        session,
        query=case["query"],
        board=case["board"],
        class_level=case["class_level"],
        subject=case["subject"],
        chapter_id=case["chapter_id"],
        chapter=case["chapter"],
        history=None,
    )
    info = panel_info(res, case["kind"])
    sc = score_case(case, res, info)
    return {
        "id": case["id"],
        "channel": channel,
        "query": case["query"],
        "ms": res.get("ms") or int((time.time() - t0) * 1000),
        "error": res.get("error"),
        "answer_clip": clip(res.get("answer"), 400),
        "panel": info,
        "score": sc,
    }


def main() -> None:
    session = AuthSession()
    print("login", getattr(session, "login_as", "?"), flush=True)
    rows: list[dict] = []
    for case in CASES:
        for channel in ("tutor", "voice"):
            print(f"\n=== {case['id']} / {channel} ===", flush=True)
            row = run_one(session, case, channel)
            rows.append(row)
            sc = row["score"]
            print(
                f"  {sc['grade']} type={sc.get('type')} defaults={sc.get('defaults')} "
                f"bugs={sc.get('bugs')} ms={row['ms']}",
                flush=True,
            )
            print(f"  ans: {row['answer_clip'][:180]}…", flush=True)

    # summary
    lines = [
        "# Live grounding examples (LLM)",
        "",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "- **Login:** Suneel",
        "- **Channels:** AI Tutor (`/auth/chat`) + AI Voice (`voice_mode` stream)",
        "",
        "## Summary",
        "",
        "| ID | Channel | Grade | Viz type | Defaults | Bugs |",
        "|----|---------|-------|----------|----------|------|",
    ]
    for r in rows:
        sc = r["score"]
        lines.append(
            f"| {r['id']} | {r['channel']} | **{sc['grade']}** | `{sc.get('type') or '—'}` | "
            f"{sc.get('defaults') or '—'} | {'; '.join(sc.get('bugs') or []) or '—'} |"
        )

    passes = sum(1 for r in rows if r["score"]["grade"] == "PASS")
    partials = sum(1 for r in rows if r["score"]["grade"] == "PARTIAL")
    fails = sum(1 for r in rows if r["score"]["grade"] == "FAIL")
    lines += [
        "",
        f"**Totals:** PASS {passes} · PARTIAL {partials} · FAIL {fails} / {len(rows)}",
        "",
        "## Details",
        "",
    ]
    for r in rows:
        lines.append(f"### {r['id']} ({r['channel']})")
        lines.append("")
        lines.append(f"- **Grade:** {r['score']['grade']}")
        lines.append(f"- **Query:** {r['query'][:200]}")
        lines.append(f"- **Answer:** {r['answer_clip']}")
        lines.append(f"- **Panel:** `{json.dumps(r['panel'], ensure_ascii=False)[:500]}`")
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    RAW.write_text(json.dumps({"rows": rows}, indent=2, default=str))
    print("\nWrote", OUT)
    print("PASS", passes, "PARTIAL", partials, "FAIL", fails)


if __name__ == "__main__":
    main()

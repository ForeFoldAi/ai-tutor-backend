#!/usr/bin/env python3
"""Focused interactive-explanation eval: 3 Qs × math/science chapters × tutor+voice.

  cd ai-tutor-backend
  .venv/bin/python suneel_chapter_qa/run_interactive_focus.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.api import AuthSession, chat_tutor, chat_voice, clip, probe_nest  # noqa: E402

BANKS = ROOT / "banks"
OUT = ROOT / "results" / "interactive_focus.md"
RAW = ROOT / "raw" / "interactive_focus.json"

# 3 questions per chapter — prefer interactive labels, else strong concept asks.
PICKS: dict[str, list[str]] = {
    "math_ch1_square_cube": ["M1-03", "M1-19", "M1-04"],
    "math_ch2_power_play": ["M2-02", "M2-11", "M2-01"],
    "science_ch1_investigative": ["S1-05", "S1-13", "S1-01"],
    "science_ch3_health": ["S3-03", "S3-19", "S3-01"],
}

# Expected viz affinity (substring of visualizationType / experimentType).
VIZ_EXPECT: dict[str, list[str]] = {
    "M1-03": ["cube", "mensuration", "shape"],
    "M1-19": ["square", "factor", "place", "mensuration", "algebra", "concept"],
    "M1-04": ["square", "factor", "place", "mensuration", "concept"],
    "M2-02": ["exponent", "power", "concept", "number"],
    "M2-11": ["exponent", "power", "concept", "number", "algebra"],
    "M2-01": ["exponent", "power", "concept", "number"],
    "S1-05": ["experiment", "concept", "microscope", "ecosystem", "lab"],
    "S1-13": ["experiment", "concept", "lab", "ecosystem"],
    "S1-01": ["concept", "experiment", "lab", "ecosystem", "microscope"],
    "S3-03": ["disease", "health", "transmission", "human", "concept", "lab"],
    "S3-19": ["disease", "germ", "transmission", "human", "microscope", "concept"],
    "S3-01": ["health", "disease", "human", "concept", "lab"],
}


def load_q(slug: str, qid: str) -> tuple[dict, dict]:
    bank = json.loads((BANKS / f"{slug}.json").read_text())
    for q in bank["questions"]:
        if q["id"] == qid:
            return bank, q
    raise KeyError(qid)


def panel_type(math_lesson, science_experiment) -> tuple[str | None, str | None]:
    if math_lesson and isinstance(math_lesson, dict):
        viz = math_lesson.get("visualization") or {}
        return "math_lesson", (viz.get("visualizationType") or math_lesson.get("visualizationType") or "")
    if science_experiment and isinstance(science_experiment, dict):
        exp = science_experiment.get("experiment") or science_experiment
        return "science_experiment", (exp.get("experimentType") or science_experiment.get("experimentType") or "")
    return None, None


def interactive_quality(
    *,
    q: dict,
    answer: str,
    math_lesson,
    science_experiment,
    channel: str,
) -> dict:
    """0–10: presence + type affinity + controls + answer keywords."""
    bugs: list[str] = []
    a = (answer or "").lower()
    hits = sum(1 for k in (q.get("expect_keywords") or []) if k.lower() in a)
    denom = max(1, len(q.get("expect_keywords") or []))
    expect = round(10.0 * hits / denom, 1)

    kind, vtype = panel_type(math_lesson, science_experiment)
    want_math = bool(q.get("want_math"))
    want_sci = bool(q.get("want_science"))

    # Voice Nest historically drops panels — score spoken teaching if no panel.
    if channel == "voice" and not kind:
        presence = 6.0 if len(a) >= 120 else 3.0
        if len(a) < 120:
            bugs.append("voice thin / no interactive panel")
        affinity = 5.0
        controls = 5.0
        note = "N/A panel (voice spoken)"
    else:
        note = ""
        if want_math:
            if kind == "math_lesson":
                presence = 9.5
            elif kind == "science_experiment":
                presence = 4.0
                bugs.append("got science panel for math Q")
            else:
                presence = 2.0
                bugs.append("math_lesson missing")
        elif want_sci:
            if kind == "science_experiment":
                presence = 9.5
            elif kind == "math_lesson":
                presence = 4.0
                bugs.append("got math panel for science Q")
            else:
                presence = 2.0
                bugs.append("science_experiment missing")
        else:
            presence = 8.0 if not kind else 6.0

        expect_bits = VIZ_EXPECT.get(q["id"], [])
        vt = (vtype or "").lower()
        if kind and expect_bits:
            if any(b in vt for b in expect_bits):
                affinity = 9.5
            elif vt in ("concept-explorer", "generic", ""):
                affinity = 5.0
                bugs.append(f"weak/generic type: {vtype or 'empty'}")
            else:
                affinity = 6.5
                bugs.append(f"type mismatch-ish: {vtype}")
        elif kind:
            affinity = 7.0
        else:
            affinity = 2.0

        controls = 2.0
        panel = math_lesson or science_experiment
        if isinstance(panel, dict):
            viz = panel.get("visualization") or panel.get("experiment") or panel
            sliders = viz.get("sliders") if isinstance(viz, dict) else None
            objs = viz.get("interactiveObjects") if isinstance(viz, dict) else None
            buttons = viz.get("buttons") if isinstance(viz, dict) else None
            views = viz.get("threeViews") if isinstance(viz, dict) else None
            score_c = 4.0
            if sliders:
                score_c += 2.5
            if objs or buttons:
                score_c += 1.5
            if views:
                score_c += 1.5
            if viz.get("liveCalculations") if isinstance(viz, dict) else None:
                score_c += 1.0
            controls = min(10.0, score_c)

    ans = 7.0 if len(a) > 80 else (3.0 if len(a) < 40 else 5.0)
    if hits:
        ans = min(10.0, ans + 1.2 * hits)
    else:
        ans = max(1.0, ans - 2.5)
        bugs.append("missing expected keywords")

    parts = [presence, affinity, controls, ans, expect]
    overall = round(sum(parts) / len(parts), 1)
    return {
        "overall": overall,
        "presence": round(presence, 1),
        "type_affinity": round(affinity, 1),
        "controls": round(controls, 1),
        "answer": round(ans, 1),
        "expect": expect,
        "panel_kind": kind,
        "viz_type": vtype or None,
        "bugs": bugs,
        "note": note,
    }


def run_one(session: AuthSession, bank: dict, q: dict, channel: str) -> dict:
    runner = chat_tutor if channel == "tutor" else chat_voice
    res = runner(
        session,
        query=q["query"],
        history=None,
        board=bank["board"],
        class_level=bank["class_level"],
        subject=bank["subject"],
        chapter_id=bank["chapter_id"],
        chapter=bank["chapter"],
    )
    sc = interactive_quality(
        q=q,
        answer=res.get("answer") or "",
        math_lesson=res.get("math_lesson"),
        science_experiment=res.get("science_experiment"),
        channel=channel,
    )
    return {
        "id": q["id"],
        "query": q["query"],
        "label": q["label"],
        "channel": channel,
        "answer_clip": clip(res.get("answer"), 350),
        "error": res.get("error"),
        "ms": res.get("ms"),
        "has_math_lesson": bool(res.get("math_lesson")),
        "has_science_experiment": bool(res.get("science_experiment")),
        "math_lesson": res.get("math_lesson"),
        "science_experiment": res.get("science_experiment"),
        "scores": sc,
    }


def avg(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return round(sum(r["scores"]["overall"] for r in rows) / len(rows), 2)


def main() -> None:
    session = AuthSession()
    print("login as", getattr(session, "login_as", "?"), flush=True)

    all_rows: list[dict] = []
    by_chapter: dict[str, dict] = {}

    for slug, ids in PICKS.items():
        bank, _ = load_q(slug, ids[0])
        nest = probe_nest(session, bank)
        print(f"\n=== {slug} nest={nest.get('ok')} ===", flush=True)
        tutor_rows: list[dict] = []
        voice_rows: list[dict] = []
        for qid in ids:
            _, q = load_q(slug, qid)
            print(f"  tutor {qid}…", flush=True)
            tr = run_one(session, bank, q, "tutor")
            tutor_rows.append(tr)
            print(f"    -> {tr['scores']['overall']} {tr['scores']['panel_kind']} {tr['scores']['viz_type']} {tr['scores']['bugs']}", flush=True)
            print(f"  voice {qid}…", flush=True)
            vr = run_one(session, bank, q, "voice")
            voice_rows.append(vr)
            print(f"    -> {vr['scores']['overall']} {vr['scores']['panel_kind']} {vr['scores']['viz_type']} {vr['scores']['bugs']}", flush=True)
        by_chapter[slug] = {
            "meta": {
                "slug": slug,
                "subject": bank["subject"],
                "chapter": bank["chapter"],
                "chapter_id": bank["chapter_id"],
            },
            "nest_probe": nest,
            "tutor": tutor_rows,
            "voice": voice_rows,
            "tutor_avg": avg(tutor_rows),
            "voice_avg": avg(voice_rows),
            "combined_avg": round((avg(tutor_rows) + avg(voice_rows)) / 2, 2),
        }
        all_rows.extend(tutor_rows)
        all_rows.extend(voice_rows)

    ranking = sorted(
        by_chapter.values(),
        key=lambda c: c["combined_avg"],
        reverse=True,
    )

    math_t = [r for r in all_rows if r["channel"] == "tutor" and r["id"].startswith("M")]
    math_v = [r for r in all_rows if r["channel"] == "voice" and r["id"].startswith("M")]
    sci_t = [r for r in all_rows if r["channel"] == "tutor" and r["id"].startswith("S")]
    sci_v = [r for r in all_rows if r["channel"] == "voice" and r["id"].startswith("S")]

    lines = [
        "# Interactive explanation focus — Suneel",
        "",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "- **Login:** Suneel",
        "- **Scope:** 3 questions × 4 chapters (2 Math + 2 Science) × AI Tutor + AI Voice",
        "- **Rubric:** presence of `math_lesson`/`science_experiment`, viz type affinity to prompt, controls (sliders/buttons/threeViews), answer keywords",
        "",
        "## Final ranks (combined Tutor+Voice)",
        "",
        "| Rank | Chapter | Subject | Tutor | Voice | Combined | Correct interactive? |",
        "|------|---------|---------|-------|-------|----------|----------------------|",
    ]
    for i, c in enumerate(ranking, 1):
        m = c["meta"]
        # Correct if tutor avg presence-ish: combined >= 7 and tutor has ≥2 panels of right kind
        tutor_ok = sum(
            1
            for r in c["tutor"]
            if (r["scores"]["panel_kind"] == "math_lesson" and m["subject"] == "Mathematics")
            or (r["scores"]["panel_kind"] == "science_experiment" and m["subject"] == "Science")
        )
        verdict = (
            "YES — reliable"
            if c["tutor_avg"] >= 7.5 and tutor_ok >= 2
            else "PARTIAL — panels flaky"
            if tutor_ok >= 1 or c["tutor_avg"] >= 6
            else "NO — missing interactive"
        )
        lines.append(
            f"| {i} | {m['chapter']} | {m['subject']} | {c['tutor_avg']} | {c['voice_avg']} | **{c['combined_avg']}** | {verdict} |"
        )

    lines += [
        "",
        "## Subject / channel rollup",
        "",
        f"| Slice | Avg /10 |",
        f"|-------|---------|",
        f"| Math · AI Tutor | **{avg(math_t)}** |",
        f"| Math · AI Voice | **{avg(math_v)}** |",
        f"| Science · AI Tutor | **{avg(sci_t)}** |",
        f"| Science · AI Voice | **{avg(sci_v)}** |",
        f"| **Overall** | **{avg(all_rows)}** |",
        "",
        "## Per-question detail",
        "",
    ]

    for c in ranking:
        m = c["meta"]
        lines.append(f"### {m['chapter']}")
        lines.append("")
        lines.append("| ID | Ch | Overall | Presence | Type | Controls | Ans | Panel | Viz type | Bugs |")
        lines.append("|----|----|---------|----------|------|----------|-----|-------|----------|------|")
        for r in c["tutor"] + c["voice"]:
            s = r["scores"]
            bugs = "; ".join(s["bugs"]) or "—"
            lines.append(
                f"| {r['id']} | {r['channel']} | {s['overall']} | {s['presence']} | {s['type_affinity']} | "
                f"{s['controls']} | {s['answer']} | {s['panel_kind'] or '—'} | `{s['viz_type'] or '—'}` | {bugs} |"
            )
        lines.append("")
        for r in c["tutor"]:
            lines.append(f"- **{r['id']} tutor:** {r['answer_clip']}")
        lines.append("")

    # Verdict block
    math_panels = sum(1 for r in math_t if r["has_math_lesson"])
    sci_panels = sum(1 for r in sci_t if r["has_science_experiment"])
    lines += [
        "## Verdict",
        "",
        f"- **Math interactive (Tutor):** {math_panels}/6 questions emitted `math_lesson`.",
        f"- **Science interactive (Tutor):** {sci_panels}/6 questions emitted `science_experiment`.",
        f"- **Math is generating correct interactive explanations** when the panel appears "
        f"({'strong' if math_panels >= 5 else 'mixed' if math_panels >= 3 else 'weak'} coverage).",
        f"- **Science interactive JSON is** "
        f"{'working' if sci_panels >= 4 else 'mostly missing' if sci_panels <= 2 else 'inconsistent'} "
        f"on Tutor — Voice often teaches in prose without forwarding the panel.",
        "",
    ]

    OUT.write_text("\n".join(lines) + "\n")
    # slim raw (drop huge nested lesson bodies partially)
    slim = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "ranking": [
            {
                "slug": c["meta"]["slug"],
                "chapter": c["meta"]["chapter"],
                "subject": c["meta"]["subject"],
                "tutor_avg": c["tutor_avg"],
                "voice_avg": c["voice_avg"],
                "combined_avg": c["combined_avg"],
            }
            for c in ranking
        ],
        "chapters": {},
    }
    def _slim_row(r: dict) -> dict:
        return {
            k: r[k]
            for k in (
                "id",
                "query",
                "label",
                "channel",
                "answer_clip",
                "error",
                "ms",
                "has_math_lesson",
                "has_science_experiment",
                "scores",
            )
        }

    for slug, c in by_chapter.items():
        slim["chapters"][slug] = {
            **{k: c[k] for k in ("meta", "nest_probe", "tutor_avg", "voice_avg", "combined_avg")},
            "tutor": [_slim_row(r) for r in c["tutor"]],
            "voice": [_slim_row(r) for r in c["voice"]],
        }
    RAW.write_text(json.dumps(slim, indent=2))
    print("\nWrote", OUT)
    print("Wrote", RAW)
    print("OVERALL", avg(all_rows))


if __name__ == "__main__":
    main()

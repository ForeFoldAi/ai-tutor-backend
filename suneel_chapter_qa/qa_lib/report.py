"""Markdown + raw JSON writers for chapter runs and the cross-chapter index."""
from __future__ import annotations

import copy
import statistics
import time
from collections import Counter
from pathlib import Path

CHANNELS = (
    ("tutor", "AI Tutor"),
    ("voice", "AI Voice"),
    ("assistant", "Ask Learn Mastro"),
)

_FIX_LOCUS = {
    "expected math_lesson missing": "`interactive_grounding.py`, math prompts",
    "expected science_experiment missing": "`interactive_grounding.py`, science prompts",
    "wanted images but got none": "`chat_service/images.py`, textbook image retrieval",
    "transcript thin": "voice_mode postprocess / `rag.client.ts`",
    "forbidden content": "`chapter_scope.py` / wrong-premise handling in `prompts.py`",
    "missing expected keywords": "retrieval (`section_retrieval.py`) / answer prompt",
    "very short answer": "answer prompt length rules / postprocess",
}


def _rows(channel_data: dict) -> list[dict]:
    return list(channel_data.get("primaries") or []) + list(channel_data.get("followups") or [])


def _pct(values: list[int], p: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def latency_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    ms = [r["ms"] for r in ok if r.get("ms")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
    overall = [r["scores"]["overall"] for r in ok if r.get("scores")]
    n = max(1, len(overall))
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "median_ms": int(statistics.median(ms)) if ms else None,
        "p90_ms": _pct(ms, 0.9),
        "mean_ms": int(statistics.mean(ms)) if ms else None,
        "ttft_median_ms": int(statistics.median(ttft)) if ttft else None,
        "ttft_p90_ms": _pct(ttft, 0.9),
        "avg": round(statistics.mean(overall), 2) if overall else 0.0,
        "ge8_pct": round(100 * sum(x >= 8 for x in overall) / n, 1),
        "poor_pct": round(100 * sum(x < 5 for x in overall) / n, 1),
    }


def _sec(ms: int | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f}s"


def slim_for_raw(payload: dict) -> dict:
    """Raw JSON keeps answer clips only (full answers bloat the file)."""
    out = copy.deepcopy(payload)
    for key, _ in CHANNELS:
        data = out.get(key) or {}
        for row in _rows(data):
            row.pop("answer", None)
    return out


def _bool(v) -> str:
    return "True" if v else "False"


def _primary_table(rows: list[dict]) -> list[str]:
    lines = [
        "| ID | Overall | Ans | Expect | Images | Interactive | n_img | math | sci | ms | TTFT | Bugs |",
        "|----|---------|-----|--------|--------|-------------|-------|------|-----|----|------|------|",
    ]
    for r in rows:
        s = r["scores"]
        inter = s["interactive_score"] if s["interactive_score"] is not None else "N/A"
        bugs = "; ".join(s["bugs"]) or "—"
        if r.get("error"):
            bugs = f"ERROR: {r['error'][:80]}"
        lines.append(
            f"| {r['id']} | {s['overall']} | {s['answer_score']} | {s['expect_score']} | "
            f"{s['images_score']} | {inter} | {r['n_images']} | {_bool(r['has_math_lesson'])} | "
            f"{_bool(r['has_science_experiment'])} | {r.get('ms')} | {r.get('ttft_ms') or '—'} | {bugs} |"
        )
    return lines


def _followup_list(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for r in rows:
        lines.append(f"- **FU ← {r['source_id']}:** asked `{r['query']}` → {r['scores']['overall']}")
        lines.append(f"  - reply: {r.get('answer_clip') or '(empty)'}")
    return lines


def write_chapter_md(path: Path, payload: dict) -> None:
    meta = payload["meta"]
    lines = [
        f"# {meta['chapter']} — Suneel chapter QA",
        "",
        f"- **Slug:** `{meta['slug']}`",
        f"- **Subject:** {meta['subject']} · **chapter_id:** `{meta['chapter_id']}` · "
        f"{meta['board']} {meta['class_level']}",
        f"- **Generated:** {payload['generated_at']}",
        f"- **Nest probe:** ok={payload.get('nest_probe', {}).get('ok')}",
    ]
    for key, title in CHANNELS:
        rows = _rows(payload.get(key) or {})
        if not rows:
            continue
        st = latency_stats(rows)
        lines.append(
            f"- **{title} avg:** **{st['avg']} / 10** · median {_sec(st['median_ms'])} · "
            f"p90 {_sec(st['p90_ms'])} · first token median {_sec(st['ttft_median_ms'])}"
        )
    qs = meta.get("questions") or []
    lines += ["", f"## Question bank ({len(qs)} primaries)", "", "| ID | Label | Query |", "|----|-------|-------|"]
    lines += [f"| {q['id']} | {q['label']} | {q['query']} |" for q in qs]

    bug_counts: Counter[str] = Counter()
    for key, title in CHANNELS:
        data = payload.get(key) or {}
        if not data.get("primaries"):
            continue
        lines += ["", f"## {title} — primaries", ""]
        lines += _primary_table(data["primaries"])
        fus = data.get("followups") or []
        lines += ["", f"## {title} — response-based follow-ups ({len(fus)})", ""]
        lines += _followup_list(fus)
        for r in _rows(data):
            for b in r["scores"]["bugs"]:
                bug_counts[f"[{key}] {b}"] += 1

    lines += ["", "## Bugs & suggested fixes", "", "| n | Signal | Suggested locus |", "|---|--------|-----------------|"]
    for sig, n in bug_counts.most_common():
        locus = next((v for k, v in _FIX_LOCUS.items() if k in sig), "—")
        lines.append(f"| {n} | {sig} | {locus} |")
    path.write_text("\n".join(lines) + "\n")


def load_summaries(raw_dir: Path, bank_index: list[dict]) -> list[dict]:
    """Per-chapter rows + stats from raw/<slug>.json, in bank order."""
    import json

    out: list[dict] = []
    for row in bank_index:
        raw = raw_dir / f"{row['slug']}.json"
        if not raw.exists():
            continue
        data = json.loads(raw.read_text())
        rows = {key: _rows(data.get(key) or {}) for key, _ in CHANNELS}
        out.append(
            {
                "slug": row["slug"],
                "subject": row["subject"],
                "chapter": row["chapter"],
                "rows": rows,
                "stats": {k: latency_stats(v) for k, v in rows.items() if v},
            }
        )
    return out


def write_index(results_dir: Path, summaries: list[dict]) -> None:
    """summaries: slug, subject, chapter, and per-channel `stats` from latency_stats()."""
    lines = [
        "# Suneel chapter QA — index",
        "",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- **Chapters:** {len(summaries)}",
        "",
        "## Scores",
        "",
        "| Chapter | Subject | Tutor avg | Voice avg | Mastro avg | MD |",
        "|---------|---------|-----------|-----------|------------|----|",
    ]

    def _avg(s: dict, key: str) -> str:
        st = (s.get("stats") or {}).get(key)
        return str(st["avg"]) if st and st.get("n") else "—"

    for s in summaries:
        lines.append(
            f"| {s['chapter']} | {s['subject']} | {_avg(s, 'tutor')} | {_avg(s, 'voice')} | "
            f"{_avg(s, 'assistant')} | [`{s['slug']}.md`]({s['slug']}.md) |"
        )

    lines += [
        "",
        "## Overall (all chapters pooled)",
        "",
        "| Channel | Turns | Errors | Avg score | ≥8 | <5 | Median total | p90 total | Median first token | p90 first token |",
        "|---------|-------|--------|-----------|----|----|--------------|-----------|--------------------|-----------------|",
    ]
    for key, title in CHANNELS:
        rows = [r for s in summaries for r in (s.get("rows") or {}).get(key, [])]
        if not rows:
            continue
        st = latency_stats(rows)
        lines.append(
            f"| {title} | {st['n']} | {st['errors']} | **{st['avg']}** | {st['ge8_pct']}% | "
            f"{st['poor_pct']}% | {_sec(st['median_ms'])} | {_sec(st['p90_ms'])} | "
            f"{_sec(st['ttft_median_ms'])} | {_sec(st['ttft_p90_ms'])} |"
        )
    lines += ["", "*Scores are heuristic (keywords, forbidden terms, images, panels) — not human-graded.*"]
    (results_dir / "index.md").write_text("\n".join(lines) + "\n")

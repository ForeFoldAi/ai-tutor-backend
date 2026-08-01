"""HTML / Markdown / JSON / console report renderers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.core.config import EvalConfig
from evaluation.core.types import RunReport, Status


def render_all(run: RunReport, cfg: EvalConfig, extra: dict[str, Any] | None = None) -> dict[str, str]:
    reports_dir = Path(cfg.get("paths", "reports")) / run.run_id
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = run.to_dict()
    if extra:
        data["overall_ai_quality"] = extra

    json_path = reports_dir / "report.json"
    md_path = reports_dir / "report.md"
    html_path = reports_dir / "report.html"
    summary_path = reports_dir / "summary.txt"

    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    md_path.write_text(to_markdown(data), encoding="utf-8")
    html_path.write_text(to_html(data), encoding="utf-8")
    summary_path.write_text(to_console(data), encoding="utf-8")
    # Convenience copies
    (Path(cfg.get("paths", "reports")) / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (Path(cfg.get("paths", "reports")) / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "console": str(summary_path),
    }


def to_console(data: dict[str, Any]) -> str:
    lines = [
        f"AI Evaluation Report — {data.get('run_id')}",
        f"Status: {data.get('overall_status')}  Score: {data.get('overall_score', 0):.3f}",
        f"Duration: {data.get('duration_ms', 0):.0f} ms",
        "",
    ]
    oq = data.get("overall_ai_quality") or {}
    if oq:
        lines.append(f"Overall AI Quality: {oq.get('overall_ai_quality_score', 0):.3f}")
        lines.append("")
    lines.append(f"{'FEATURE':<22} {'STATUS':<8} PASS FAIL  TIME(ms)")
    lines.append("-" * 60)
    for p in data.get("phases", []):
        lines.append(
            f"{p.get('feature', p.get('phase_id')):<22} {p.get('status'):<8} "
            f"{p.get('passed', 0):>4} {p.get('failed', 0):>4}  {p.get('execution_ms', 0):>8.0f}"
        )
    reg = data.get("regression") or {}
    if reg.get("regressed"):
        lines.append("")
        lines.append("REGRESSIONS:")
        for r in reg.get("regressions", []):
            lines.append(f"  - {r}")
    # Failures detail
    lines.append("")
    lines.append("FAILURES:")
    any_fail = False
    for p in data.get("phases", []):
        for c in p.get("checks", []):
            if c.get("status") in ("FAIL", "ERROR"):
                any_fail = True
                lines.append(
                    f"  [{p.get('phase_id')}] {c.get('check_id')}: {c.get('reason')}"
                )
                if c.get("expected") is not None:
                    lines.append(f"      expected={c.get('expected')}")
                if c.get("actual") is not None:
                    lines.append(f"      actual={c.get('actual')}")
    if not any_fail:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


def to_markdown(data: dict[str, Any]) -> str:
    lines = [
        f"# AI Evaluation Report",
        "",
        f"- **Run ID:** `{data.get('run_id')}`",
        f"- **Status:** **{data.get('overall_status')}**",
        f"- **Score:** {data.get('overall_score', 0):.3f}",
        f"- **Duration:** {data.get('duration_ms', 0):.0f} ms",
        "",
        "## Phases",
        "",
        "| Feature | Status | Pass | Fail | Time (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for p in data.get("phases", []):
        lines.append(
            f"| {p.get('feature')} | {p.get('status')} | {p.get('passed', 0)} | "
            f"{p.get('failed', 0)} | {p.get('execution_ms', 0):.0f} |"
        )
    lines += ["", "## Failed checks", ""]
    for p in data.get("phases", []):
        fails = [c for c in p.get("checks", []) if c.get("status") in ("FAIL", "ERROR")]
        if not fails:
            continue
        lines.append(f"### {p.get('phase_id')}")
        for c in fails:
            lines.append(f"- `{c.get('check_id')}` — {c.get('reason')}")
            lines.append(f"  - expected: `{c.get('expected')}`")
            lines.append(f"  - actual: `{c.get('actual')}`")
            if c.get("evidence"):
                lines.append(f"  - evidence: `{c.get('evidence')}`")
    reg = data.get("regression") or {}
    if reg:
        lines += ["", "## Regression", "", f"```json\n{json.dumps(reg, indent=2)}\n```"]
    return "\n".join(lines) + "\n"


def to_html(data: dict[str, Any]) -> str:
    rows = []
    for p in data.get("phases", []):
        color = {
            "PASS": "#126b3a",
            "FAIL": "#a11",
            "ERROR": "#7a0010",
            "SKIP": "#666",
            "WARN": "#8a6d00",
        }.get(p.get("status"), "#333")
        rows.append(
            f"<tr><td>{p.get('feature')}</td><td style='color:{color};font-weight:700'>"
            f"{p.get('status')}</td><td>{p.get('passed', 0)}</td><td>{p.get('failed', 0)}</td>"
            f"<td>{p.get('execution_ms', 0):.0f}</td></tr>"
        )
    fail_blocks = []
    for p in data.get("phases", []):
        for c in p.get("checks", []):
            if c.get("status") not in ("FAIL", "ERROR"):
                continue
            fail_blocks.append(
                f"<li><code>{p.get('phase_id')}.{c.get('check_id')}</code> — {c.get('reason')}"
                f"<br/><small>expected={_esc(c.get('expected'))} | actual={_esc(c.get('actual'))}</small></li>"
            )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>AI Evaluation {data.get('run_id')}</title>
<style>
body{{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;margin:2rem;background:#f7f5f1;color:#1b1b1b}}
h1{{font-size:1.6rem}} table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{border:1px solid #ddd;padding:.5rem .6rem;text-align:left}} th{{background:#eee}}
.badge{{display:inline-block;padding:.2rem .5rem;border-radius:4px;background:#222;color:#fff}}
</style></head><body>
<h1>AI Evaluation Report</h1>
<p><span class="badge">{data.get('overall_status')}</span>
 Score {data.get('overall_score', 0):.3f} · Run <code>{data.get('run_id')}</code></p>
<table><thead><tr><th>Feature</th><th>Status</th><th>Pass</th><th>Fail</th><th>Time</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Failures</h2>
<ul>{''.join(fail_blocks) or '<li>None</li>'}</ul>
</body></html>
"""


def _esc(v: Any) -> str:
    s = str(v)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:500]

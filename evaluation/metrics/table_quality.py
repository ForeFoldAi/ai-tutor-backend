"""Table extraction quality checks."""

from __future__ import annotations

import re
from typing import Any


def parse_markdown_table(text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return {"rows": 0, "cols": 0, "headers": [], "ok": False, "reason": "no_table"}
    def split_row(ln: str) -> list[str]:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        return parts

    headers = split_row(lines[0])
    # skip separator
    body_start = 2 if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", lines[1]) else 1
    body = [split_row(ln) for ln in lines[body_start:]]
    cols = len(headers)
    ragged = any(len(r) != cols for r in body)
    return {
        "rows": len(body),
        "cols": cols,
        "headers": headers,
        "ragged": ragged,
        "ok": cols > 0 and len(body) > 0 and not ragged,
        "reason": "ok" if (cols > 0 and body and not ragged) else "structure_issue",
    }


def score_table(structured_text: str, golden: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed = parse_markdown_table(structured_text)
    issues: list[str] = []
    if not parsed["ok"]:
        issues.append(parsed["reason"])
    if golden:
        if "rows" in golden and parsed["rows"] != int(golden["rows"]):
            issues.append("row_count_mismatch")
        if "cols" in golden and parsed["cols"] != int(golden["cols"]):
            issues.append("col_count_mismatch")
        if "headers" in golden:
            g_headers = [h.lower() for h in golden["headers"]]
            a_headers = [h.lower() for h in parsed["headers"]]
            if g_headers != a_headers:
                # soft: allow subset
                if not set(g_headers).issubset(set(a_headers)):
                    issues.append("header_mismatch")
    score = 1.0 - 0.25 * len(issues)
    return {
        "score": max(0.0, score),
        "issues": issues,
        "parsed": parsed,
    }

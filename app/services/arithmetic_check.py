"""Correct wrong arithmetic claims in a tutor answer before the student keeps it.

The model is reliable at explaining but not at arithmetic inside an analogy: asked to explain
2⁵ "more simply" it wrote "2⁵ = 64" (Student-understanding eval, 2026-11-23). A wrong number in
an explanation teaches the wrong thing, so simple, fully-numeric claims are recomputed:

  power claims     2⁵ = 64 · 2^5 is 64 · 2 to the power 5 equals 64 · 2**5 = 64
  product chains   2 × 2 × 2 × 2 × 2 = 64 · 3 x 3 x 3 = 9

Only exact integer claims are touched; anything ambiguous is left alone.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
# A plain integer that is not part of 1,000 / 2.5 (those are left alone).
_NUM = r"(?<![\d,.])\d{1,6}(?![.,]?\d)"
_RESULT = r"(\**\s*)(\d{1,3}(?:,\d{2,3})+|\d{1,12})(\s*\**)"
_EQ = r"\s*(?:=|is|equals|gives(?:\s+you)?|makes)\s*"

_POWER_RE = re.compile(
    rf"\b({_NUM})\s*(?:\^|\*\*|([⁰¹²³⁴⁵⁶⁷⁸⁹]+)|\s+to\s+the\s+power(?:\s+of)?\s+)\s*({_NUM})?"
    rf"(\**){_EQ}{_RESULT}",
    re.I,
)
_PRODUCT_RE = re.compile(
    rf"\b({_NUM}(?:\s*[×✕*]\s*{_NUM}|\s+x\s+{_NUM})+)\s*=\s*{_RESULT}",
    re.I,
)

_MAX_EXP = 30


def _as_int(text: str) -> int:
    return int(text.replace(",", ""))


def _fix_power(m: re.Match[str]) -> str:
    base = int(m.group(1))
    exp_txt = (m.group(2) or "").translate(_SUP) or (m.group(3) or "")
    if not exp_txt.isdigit():
        return m.group(0)
    exp = int(exp_txt)
    if exp > _MAX_EXP:
        return m.group(0)
    right = base ** exp
    claimed = _as_int(m.group(6))
    if claimed == right:
        return m.group(0)
    logger.info("arithmetic_check: %s^%s claimed %s, corrected to %s", base, exp, claimed, right)
    start, end = m.span(6)
    s0 = m.start(0)
    whole = m.group(0)
    return whole[: start - s0] + str(right) + whole[end - s0 :]


def _fix_product(m: re.Match[str]) -> str:
    factors = [int(x) for x in re.findall(r"\d+", m.group(1))]
    right = 1
    for f in factors:
        right *= f
    claimed = _as_int(m.group(3))
    if claimed == right:
        return m.group(0)
    logger.info("arithmetic_check: %s claimed %s, corrected to %s", m.group(1), claimed, right)
    start, end = m.span(3)
    s0 = m.start(0)
    whole = m.group(0)
    return whole[: start - s0] + str(right) + whole[end - s0 :]


def fix_arithmetic_claims(text: str) -> str:
    """Return ``text`` with wrong power / product results replaced by the correct value."""
    if not text or not re.search(r"\d", text):
        return text
    try:
        out = _POWER_RE.sub(_fix_power, text)
        return _PRODUCT_RE.sub(_fix_product, out)
    except Exception as exc:  # never break an answer over a checker bug
        logger.warning("arithmetic_check failed: %s", exc)
        return text


_ABBREV_RE = re.compile(r"\b(?:fig|figs|e\.g|i\.e|etc|dr|mr|mrs|ms|no|vs|ch|pg|p|approx)\.$", re.I)
_SENTENCE_END_RE = re.compile(r"[.!?…)\]\"'”’*`$]\s*$")


def trim_incomplete_tail(text: str) -> str:
    """Cut an answer that stops mid-sentence (token limit) back to its last full sentence.

    "…**Quick check**: If you look at Fig. 2.30 (a Vijayan" → ends at the previous sentence.
    Left alone when the cut would drop more than 40% of the answer.
    """
    t = (text or "").rstrip()
    if not t or _SENTENCE_END_RE.search(t) or t.endswith("```"):
        return text
    ends = [
        m.end() for m in re.finditer(r"[.!?](?=\s|$)|[.!?]\*\*(?=\s|$)", t)
        if not _ABBREV_RE.search(t[: m.start() + 1])
    ]
    if not ends:
        return text
    cut = t[: ends[-1]].rstrip()
    # Drop a dangling list bullet / heading left at the end.
    if len(cut) < 0.6 * len(t):
        return text
    return cut

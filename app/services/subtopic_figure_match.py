"""
Generic subtopic ↔ figure matching for any subject textbook.

Uses PDF text layer, title fuzzy match, and BGE semantic similarity —
no subject- or chapter-specific keyword tables.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.section_heading import (
    SubtopicInfo,
    figure_number_matches,
    is_sidebar_figure_caption,
    normalize_title,
    title_match_score,
)


def is_panel_subfigure(fig: str | None) -> bool:
    """Mini-panel figures (e.g. 2.4.1, 3.2.3) grouped under one parent figure."""
    if not fig:
        return False
    parts = str(fig).strip().split(".")
    return len(parts) >= 3 and parts[-1].isdigit()


def primary_figure_numbers(sub: SubtopicInfo) -> list[str]:
    return [fn for fn in (sub.figure_numbers or []) if fn and not is_panel_subfigure(fn)]


def is_pedagogy_activity_text(text: str) -> bool:
    """
    True when the lines immediately above the figure label are from an activity box.

    Pedagogy headings elsewhere on the same page (e.g. THINK ABOUT IT mid-page) are ignored.
    """
    from app.services.section_heading import is_pedagogy_box_text

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    tail = "\n".join(lines[-10:])
    return is_pedagogy_box_text(tail)


def pdf_context_parts(im: Any) -> tuple[str, str, str]:
    """Return (text_before_fig, text_from_fig_line, pdf_caption_line)."""
    from app.services.image_service.pdf_figure_context import (
        best_caption_line_from_pdf,
        page_text_near_figure,
        pdf_path_for_image,
    )

    pdf_path = pdf_path_for_image(im)
    fig = getattr(im, "figure_number", None)
    if not pdf_path or not fig:
        return "", "", ""

    page_idx = int(getattr(im, "page_index", 0) or 0)
    before, after = page_text_near_figure(pdf_path, page_idx, str(fig))
    pdf_cap = best_caption_line_from_pdf(pdf_path, page_idx, str(fig))
    return before, after, pdf_cap


def is_descriptive_caption_line(caption_line: str, figure_number: str | None) -> bool:
    cap = (caption_line or "").strip()
    if not cap or not figure_number:
        return False
    fig = str(figure_number).strip()
    if not re.match(rf"(?i)^fig\.?\s*{re.escape(fig)}", cap):
        return False
    return len(cap) > len(f"Fig. {fig}") + 4


def _subtopic_tokens_overlap(sub: SubtopicInfo, text: str) -> bool:
    q_tokens = set(re.findall(r"[a-z0-9]+", normalize_title(sub.title)))
    t_tokens = set(re.findall(r"[a-z0-9]+", normalize_title(text)))
    if not q_tokens:
        return False
    return bool(q_tokens & t_tokens)


def subtopic_relevance_score(im: Any, sub: SubtopicInfo) -> float:
    """
    How well figure context aligns with the textbook subtopic title (0–100).
    """
    before, _after, pdf_cap = pdf_context_parts(im)
    cap = (getattr(im, "caption", None) or "").strip()
    blob = " ".join(
        p for p in (before, pdf_cap, cap, getattr(im, "figure_context", None) or "") if p
    ).strip()

    if not blob:
        return 0.0
    if is_pedagogy_activity_text(before):
        return 0.0
    if is_sidebar_figure_caption(cap) or is_sidebar_figure_caption(blob[:300]):
        return 0.0

    title_sc = title_match_score(sub.title, blob[:2500])
    primary_text = " ".join(p for p in (pdf_cap, before[:1200]) if p)
    if not _subtopic_tokens_overlap(sub, primary_text):
        title_sc = min(title_sc, 40.0)

    try:
        from app.services.image_service.figure_context_bge import figure_context_bge_score

        bge_sc = figure_context_bge_score(sub.title, im)
    except Exception:
        bge_sc = 0.0

    if bge_sc > 0 and title_sc >= 45:
        return max(title_sc, bge_sc * 0.85)
    if title_sc >= 45:
        return title_sc
    if bge_sc >= 62 and _subtopic_tokens_overlap(sub, primary_text):
        return bge_sc * 0.85
    return min(title_sc, bge_sc)


def score_image_for_subtopic(im: Any, sub: SubtopicInfo) -> float:
    """Rank one textbook figure for a lettered subtopic (higher = better)."""
    cap_only = (getattr(im, "caption", None) or "").strip()
    if is_sidebar_figure_caption(cap_only):
        return -1.0

    before, _after, pdf_cap = pdf_context_parts(im)
    fig = getattr(im, "figure_number", None)
    all_figs = sub.figure_numbers or []
    primary_figs = primary_figure_numbers(sub)
    authoritative = bool(fig and all_figs and figure_number_matches(fig, all_figs))

    if not authoritative and is_pedagogy_activity_text(before):
        return -1.0

    page_idx = int(getattr(im, "page_index", 0) or 0)
    if sub.pages and page_idx > max(sub.pages) + 1:
        return -1.0

    # Trust Fig numbers parsed from the textbook subtopic block (a) b) c) …).
    if authoritative:
        score = 92.0
        if primary_figs and figure_number_matches(fig, primary_figs):
            score += 28.0
            for idx, pf in enumerate(primary_figs):
                if figure_number_matches(fig, [pf]):
                    score += max(0, 14 - idx * 5)
                    break
        blob = f"{pdf_cap} {cap_only}".lower()
        title_tokens = set(re.findall(r"[a-z]{4,}", normalize_title(sub.title)))
        if title_tokens & set(re.findall(r"[a-z]{4,}", blob)):
            score += 16.0
        if is_descriptive_caption_line(pdf_cap, fig):
            score += 18.0
        if sub.pages:
            anchor = max(sub.pages)
            dist = abs(page_idx - anchor)
            score += max(0.0, 30.0 - dist * 10.0)
        sub_key = sub.normalized_title
        for field in ("subsection_title", "section_title"):
            val = getattr(im, field, None) or ""
            if sub_key and sub_key in normalize_title(val):
                score += 12.0
        score += float(getattr(im, "educational_salience", 0.5) or 0.5) * 5.0
        return score

    relevance = subtopic_relevance_score(im, sub)
    if relevance < 45.0:
        return -1.0

    score = relevance
    if fig and is_panel_subfigure(fig) and primary_figs:
        return -1.0

    match_figs = primary_figs or all_figs
    if match_figs and figure_number_matches(fig, match_figs):
        score += 80.0
    if is_descriptive_caption_line(pdf_cap, fig):
        score += 30.0

    if sub.pages:
        anchor = max(sub.pages)
        dist = abs(page_idx - anchor)
        score += max(0.0, 35.0 - dist * 10.0)

    sub_key = sub.normalized_title
    for field in ("subsection_title", "section_title"):
        val = getattr(im, field, None) or ""
        if sub_key and sub_key in normalize_title(val):
            score += 40.0

    if getattr(im, "is_decorative", False):
        score -= 40.0
    score += float(getattr(im, "educational_salience", 0.5) or 0.5) * 10.0
    return score

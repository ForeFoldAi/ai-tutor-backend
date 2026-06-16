"""
Backfill figure surrounding text from the PDF text layer.

ML layout crops often produce OCR-fragment captions; the native PDF text
layer still has the real definition paragraphs (e.g. Fig 2.2 + "Weather is…").
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

_FIG_LINE_RE = re.compile(r"(?i)\bfig\.?\s*(\d+(?:\.\d+)*)")


@lru_cache(maxsize=256)
def _page_lines(pdf_path: str, page_index: int) -> tuple[str, ...]:
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            if page_index < 0 or page_index >= len(doc):
                return ()
            return tuple(doc[page_index].get_text().splitlines())
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("pdf page read failed %s p%d: %s", pdf_path, page_index, exc)
        return ()


def page_text_near_figure(
    pdf_path: str,
    page_index: int,
    figure_number: str | None,
    *,
    lines_before: int = 18,
    lines_after: int = 6,
) -> tuple[str, str]:
    """
    Return (text_before_fig_line, text_from_fig_line_onward) from PDF text layer.
    """
    if not pdf_path or not figure_number:
        return "", ""

    lines = list(_page_lines(pdf_path, page_index))
    if not lines:
        return "", ""

    fig_idx = None
    pat = re.compile(rf"(?i)\bfig\.?\s*{re.escape(figure_number)}\b")
    for i, line in enumerate(lines):
        if pat.search(line):
            fig_idx = i
            break

    if fig_idx is None:
        return "", ""

    # NCERT pages often place the definition at the top and the figure at the bottom.
    if fig_idx >= 8:
        before = "\n".join(ln.strip() for ln in lines[:fig_idx] if ln.strip())
    else:
        before = "\n".join(
            ln.strip() for ln in lines[max(0, fig_idx - lines_before) : fig_idx] if ln.strip()
        )
    after = "\n".join(ln.strip() for ln in lines[fig_idx : min(len(lines), fig_idx + lines_after)] if ln.strip())
    return before, after


def page_supports_core_definition(pdf_path: str, page_index: int, core_concept: str) -> bool:
    """True when the PDF page contains a textbook definition of *core_concept*."""
    lines = list(_page_lines(pdf_path, page_index))
    if not lines:
        return False
    text = "\n".join(lines).lower()
    core = (core_concept or "").strip().lower()
    if not core:
        return False
    if re.search(rf"(?i)\bwhat\s+is\s+{re.escape(core)}\b", text):
        return True
    return bool(re.search(rf"(?i)\b{re.escape(core)}\s+is\s+(?:a|an|the)\s+", text))


def caption_line_from_pdf(
    pdf_path: str,
    page_index: int,
    figure_number: str | None,
) -> str:
    """Best caption line for a figure from PDF text (Fig. N + title if present)."""
    if not pdf_path or not figure_number:
        return ""

    lines = list(_page_lines(pdf_path, page_index))
    pat = re.compile(rf"(?i)\bfig\.?\s*{re.escape(figure_number)}\b")
    fallback = ""
    for line in lines:
        if not pat.search(line):
            continue
        cleaned = re.sub(r"\s+", " ", line.strip())
        if len(cleaned) < 6:
            continue
        if re.match(rf"(?i)^fig\.?\s*{re.escape(figure_number)}", cleaned):
            return cleaned[:500]
        if not fallback:
            fallback = cleaned
    return fallback[:500]


def best_caption_line_from_pdf(
    pdf_path: str,
    page_index: int,
    figure_number: str | None,
) -> str:
    """Caption line from the figure page and adjacent spread pages."""
    if not pdf_path or not figure_number:
        return ""

    best = caption_line_from_pdf(pdf_path, page_index, figure_number)
    fig = str(figure_number).strip()
    if best and re.match(rf"(?i)^fig\.?\s*{re.escape(fig)}", best.strip()):
        if len(best.strip()) > len(f"Fig. {fig}") + 4:
            return best

    for adj in (page_index - 1, page_index + 1):
        if adj < 0:
            continue
        alt = caption_line_from_pdf(pdf_path, adj, figure_number)
        if alt and re.match(rf"(?i)^fig\.?\s*{re.escape(fig)}", alt.strip()):
            if len(alt.strip()) > len(f"Fig. {fig}") + 4:
                return alt
    return best


def pdf_path_for_image(im) -> str | None:
    upload = getattr(im, "upload", None)
    if upload is not None:
        path = getattr(upload, "file_path", None)
        if path:
            return str(path)
    return None


_REPRINT_NOISE_RE = re.compile(r"(?i)\breprint\s+20\d{2}[-/]?\d*\b.*$")
_CAPTION_FIG_PREFIX_RE = re.compile(
    r"(?i)^(?:fig\.?|figure|diagram|illustration|plate)\s*\d+(?:\.\d+)*\s*[.:;\-–—]?\s*"
)


def _caption_needs_backfill(caption: str | None) -> bool:
    from app.services.image_service.figure_context_gates import (
        is_corrupt_ml_caption,
        is_minimal_figure_caption,
    )

    cap = (caption or "").strip()
    if not cap:
        return True
    if is_corrupt_ml_caption(cap) or is_minimal_figure_caption(cap):
        return True
    if re.fullmatch(r"\d{1,3}", cap):
        return True
    if len(cap) < 12:
        return True
    return False


_PROSE_CAPTION_RE = re.compile(
    r"\b(you wake up|what is|chapter \d|grade \d|exploring society|india and beyond|"
    r"weather and its elements|keep you cool|keep yourself warm|reach for thick)\b|"
    r"^in th[e]?\s+[a-z]",
    re.I,
)


def _short_figure_label(text: str, *, max_words: int = 14) -> str:
    """Keep only a brief figure caption (one short phrase), never body paragraphs."""
    from app.services.image_service.figure_context_gates import is_corrupt_ml_caption

    label = _CAPTION_FIG_PREFIX_RE.sub("", (text or "").strip()).strip(" .:;-")
    # Drop leading type prefixes from generated captions.
    label = re.sub(r"(?i)^(illustration|diagram|photograph|map|chart)\s*[—\-:]\s*", "", label).strip()
    label = _REPRINT_NOISE_RE.sub("", label).strip(" .")
    if not label or is_corrupt_ml_caption(label):
        return ""
    if _PROSE_CAPTION_RE.search(label):
        return ""
    words = label.split()
    if len(words) > max_words:
        label = " ".join(words[:max_words]).rstrip(".,;:") + "…"
    if len(label) > 140:
        label = label[:137].rsplit(" ", 1)[0] + "…"
    return label


def _is_minimal_figure_line(text: str, figure_number: str | None) -> bool:
    """True when the line is only 'Fig. 2.2' with no descriptive label."""
    if not text or not figure_number:
        return not (text or "").strip()
    label = _CAPTION_FIG_PREFIX_RE.sub("", text.strip()).strip(" .:;-")
    return len(label) < 4


def _definition_before_figure(before: str) -> str:
    """Pick a short textbook definition sentence that appears above the figure."""
    text = re.sub(r"\s+", " ", (before or "")).strip()
    if not text:
        return ""

    patterns = [
        r"(?i)weather is a state of the earth['\u2019]?s atmosphere at a particular time and place\.?",
        r"(?i)weather is a state of[^.?!]{5,90}[.?!]",
        r"(?i)(?:climate|precipitation|humidity|temperature|wind|atmosphere) is (?:a|an|the)[^.?!]{8,100}[.?!]",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if not match:
            continue
        sent = match.group(0).strip()
        # Keep weather definitions to one tight phrase for the UI.
        if re.search(r"(?i)^weather is a state of", sent):
            sent = re.split(r"(?i)\s+at a particular\b", sent, maxsplit=1)[0].strip(" .")
        label = _short_figure_label(sent, max_words=12)
        if label:
            return label
    return ""


def _caption_from_nearby_page(
    im,
    pdf_path: str,
    page_index: int,
    figure_number: str,
    *,
    subtopic: str | None = None,
) -> str:
    """Build a short caption when the PDF figure line has no title."""
    from app.services.image_service.caption_generator import _best_sentence, _concept_from_headings

    before, after = page_text_near_figure(pdf_path, page_index, figure_number)

    definition = _definition_before_figure(before)
    if definition:
        return definition

    concept = _concept_from_headings(
        getattr(im, "section_title", None),
        getattr(im, "subsection_title", None),
        getattr(im, "chapter_title", None) or subtopic,
    )
    best = _best_sentence(before, after, concept)
    if best:
        label = _short_figure_label(best)
        if label:
            return label
    return ""


def resolve_display_caption(im, *, subtopic: str | None = None) -> str:
    """
    Short student-facing caption (label only — figure number is sent separately).
    """
    from app.services.image_service.figure_context_gates import is_corrupt_ml_caption

    raw = (getattr(im, "caption", None) or getattr(im, "title", None) or "").strip()

    pdf_path = pdf_path_for_image(im)
    fig = getattr(im, "figure_number", None)
    page_idx = int(getattr(im, "page_index", 0) or 0)

    if pdf_path and fig:
        pdf_cap = best_caption_line_from_pdf(pdf_path, page_idx, str(fig))
        label = _short_figure_label(pdf_cap)
        if label:
            return label
        if _is_minimal_figure_line(pdf_cap, fig) or _is_minimal_figure_line(raw, fig):
            nearby = _caption_from_nearby_page(
                im, pdf_path, page_idx, str(fig), subtopic=subtopic
            )
            if nearby:
                return nearby

    raw_label = _short_figure_label(raw)
    if raw_label:
        return raw_label

    if pdf_path and fig:
        nearby = _caption_from_nearby_page(im, pdf_path, page_idx, str(fig), subtopic=subtopic)
        if nearby:
            return nearby

    if subtopic and fig and not is_corrupt_ml_caption(subtopic):
        return subtopic[:120]

    chapter = (getattr(im, "chapter_title", None) or "").strip()
    if chapter and fig:
        topic = re.sub(r"(?i)^chapter\s+\d+\s*[-–—:]\s*", "", chapter).strip()
        if topic and len(topic) >= 4:
            return topic[:120]

    if fig:
        return "Textbook illustration"
    return raw[:120] if raw and not _caption_needs_backfill(raw) else "Textbook illustration"

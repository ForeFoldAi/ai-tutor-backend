"""
Math-textbook extraction helpers: formula bbox padding, PDF text fallback, decorative filters.
"""

from __future__ import annotations

import re

import fitz

from app.services.image_service.formula_bbox import expand_formula_bbox

__all__ = [
    "expand_formula_bbox",
    "extract_formula_text_from_pdf",
    "is_decorative_math_figure",
    "is_oversized_formula_prose_crop",
    "looks_like_math_line",
    "merge_formula_structured_text",
    "section_heading_above_figure",
]

_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9 ,\-/]{3,70})$")
_MATH_CHAR_RE = re.compile(r"[=+\-×÷*/^\\()\[\]{}]")
_DECORATIVE_MATH_RE = re.compile(
    r"(?i)\b("
    r"how did they|how do they|do you know|think about|curious about|"
    r"would you like|let us explore|speech bubble|cartoon"
    r")\b",
)
_SPEECH_QUESTION_RE = re.compile(r"(?i)^(how|what|why|when|did|can)\b.+\?\s*$")


def text_from_pdf_region(
    page: fitz.Page,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
) -> str:
    from app.services.pdf_extract_pipeline.raster import image_box_to_pdf_rect

    rect = image_box_to_pdf_rect(box, page.rect, image_size, dpi)
    if rect.is_empty:
        return ""
    return (page.get_text("text", clip=rect) or "").strip()


def looks_like_math_line(line: str) -> bool:
    s = (line or "").strip()
    if len(s) < 2:
        return False
    if _MATH_CHAR_RE.search(s):
        alpha = len(re.findall(r"[A-Za-z]", s))
        return alpha <= max(24, len(s) // 2)
    return False


def extract_formula_text_from_pdf(
    page: fitz.Page,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
) -> str:
    """Pull selectable math lines from the PDF text layer inside (padded) formula region."""
    expanded = expand_formula_bbox(box, image_size)
    raw = text_from_pdf_region(page, expanded, image_size, dpi)
    if not raw:
        return ""
    math_lines = [ln.strip() for ln in raw.splitlines() if looks_like_math_line(ln.strip())]
    return "\n".join(math_lines).strip()


def merge_formula_structured_text(pdf_text: str, latex: str) -> str:
    """Prefer complete PDF text when LaTeX is missing or clearly truncated."""
    pdf_text = (pdf_text or "").strip()
    latex = (latex or "").strip()
    if not pdf_text:
        return latex
    if not latex:
        return pdf_text
    if len(latex) < 8 and len(pdf_text) > len(latex):
        return pdf_text
    if len(pdf_text) > len(latex) * 1.35:
        return pdf_text
    return latex


def is_oversized_formula_prose_crop(
    structured_text: str,
    *,
    width: int,
    height: int,
) -> bool:
    """Reject formula crops that are mostly explanatory paragraphs."""
    area = max(1, width * height)
    words = len(re.findall(r"\b\w+\b", structured_text or ""))
    if area > 180_000 and words > 35:
        return True
    if height > 220 and width > 500 and words > 25:
        return True
    return False


def is_decorative_math_figure(
    *,
    caption: str,
    figure_context: str,
    figure_number: str | None,
    width: int,
    height: int,
) -> bool:
    """Filter speech-bubble cartoons and tiny unlabeled clipart from math chapters."""
    cap = (caption or "").strip()
    ctx = (figure_context or "").strip()
    blob = f"{cap} {ctx}"

    if figure_number:
        return False

    if _DECORATIVE_MATH_RE.search(blob):
        return True

    if cap and _SPEECH_QUESTION_RE.match(cap):
        return True

    if cap.endswith("?") and len(cap.split()) >= 6 and not re.search(r"(?i)\bfig", cap):
        return True

    if width > 0 and height > 0:
        if width < 130 and height < 130 and not cap:
            return True
        aspect = width / max(1, height)
        if aspect > 2.8 and height < 180 and "?" in cap:
            return True

    return False


def section_heading_above_figure(
    page: fitz.Page,
    figure_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
) -> str:
    """Best-effort section title above an unnumbered figure (Large Numbers chapters)."""
    xmin, ymin, xmax, ymax = figure_box
    above_box = (
        max(0, xmin - 60),
        max(0, ymin - 520),
        min(image_size[0], xmax + 60),
        ymin,
    )
    text = text_from_pdf_region(page, above_box, image_size, dpi)
    best = ""
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not line or len(line) > 80:
            continue
        m = _NUMBERED_HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
        words = line.split()
        if 2 <= len(words) <= 10 and sum(1 for w in words if w and w[0].isupper()) / len(words) >= 0.7:
            if not line.endswith("."):
                best = best or line
    return best

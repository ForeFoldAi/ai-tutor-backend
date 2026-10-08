"""Document profiling and adaptive PDF characteristic extraction."""

from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class DocumentProfile:
    total_pages: int
    pdf_type: str  # "text_based", "scanned", "mixed", "low_extraction_quality"
    baseline_font_size: float
    has_native_bookmarks: bool
    bookmark_count: int
    is_landscape: bool
    page_width: float
    page_height: float
    avg_chars_per_page: float
    image_only_pages: list[int] = field(default_factory=list)
    repeated_headers: list[str] = field(default_factory=list)
    repeated_footers: list[str] = field(default_factory=list)
    detected_watermarks: list[str] = field(default_factory=list)
    margin_page_number_position: str = "none"  # "top", "bottom", "none"


def profile_document(doc: fitz.Document) -> DocumentProfile:
    """Analyze a PDF document to extract structural, typographic, and layout characteristics."""
    total_pages = len(doc)
    if total_pages == 0:
        return DocumentProfile(
            total_pages=0,
            pdf_type="text_based",
            baseline_font_size=10.0,
            has_native_bookmarks=False,
            bookmark_count=0,
            is_landscape=False,
            page_width=0.0,
            page_height=0.0,
            avg_chars_per_page=0.0,
        )

    # 1. Sample pages across document for statistical evaluation
    sample_indices = list(range(min(15, total_pages)))
    if total_pages > 15:
        step = max(1, total_pages // 15)
        sample_indices.extend(range(15, total_pages, step))
    sample_indices = sorted(list(set(sample_indices)))

    total_chars = 0
    low_text_pages = []
    image_only_pages = []
    font_sizes: list[float] = []
    top_margin_lines: list[str] = []
    bottom_margin_lines: list[str] = []
    full_page_blocks: list[str] = []
    top_digits_count = 0
    bottom_digits_count = 0

    first_page = doc[0]
    p_width, p_height = first_page.rect.width, first_page.rect.height
    is_landscape = p_width > p_height

    for pno in sample_indices:
        page = doc[pno]
        text = page.get_text() or ""
        char_len = len(text.strip())
        total_chars += char_len

        # Check for image presence without text
        images = page.get_images()
        if char_len < 50:
            low_text_pages.append(pno)
            if images:
                image_only_pages.append(pno)

        # Margin and watermark analysis via blocks and dict
        h = page.rect.height
        w = page.rect.width
        top_rect = fitz.Rect(0, 0, w, min(70, h * 0.12))
        bot_rect = fitz.Rect(0, max(0, h - 75), w, h)

        top_txt = page.get_text("text", clip=top_rect).strip()
        bot_txt = page.get_text("text", clip=bot_rect).strip()

        for ln in top_txt.split("\n"):
            ln_s = ln.strip()
            if len(ln_s) >= 3 and not ln_s.isdigit():
                top_margin_lines.append(ln_s)
            elif ln_s.isdigit():
                top_digits_count += 1

        for ln in bot_txt.split("\n"):
            ln_s = ln.strip()
            if len(ln_s) >= 3 and not ln_s.isdigit():
                bottom_margin_lines.append(ln_s)
            elif ln_s.isdigit():
                bottom_digits_count += 1

        # Check blocks for large diagonal watermarks or background stamps
        blocks = page.get_text("blocks")
        for b in blocks:
            b_txt = b[4].strip()
            b_w = b[2] - b[0]
            b_h = b[3] - b[1]
            # If block covers more than 60% of page width and 60% of page height, it's likely a watermark/background
            if b_w >= 0.6 * w and b_h >= 0.5 * h and len(b_txt.split("\n")) <= 3:
                clean_b = b_txt.replace("\n", " ").strip()
                if len(clean_b) >= 4:
                    full_page_blocks.append(clean_b)

        # Font size distribution
        page_dict = page.get_text("dict")
        for b in page_dict.get("blocks", []):
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    st = span.get("text", "").strip()
                    if len(st) > 4:
                        font_sizes.append(round(span.get("size", 10.0), 1))

    # 2. Determine document classification
    sample_count = max(1, len(sample_indices))
    low_text_ratio = len(low_text_pages) / sample_count
    avg_chars = total_chars / sample_count

    if low_text_ratio >= 0.85:
        pdf_type = "scanned"
    elif low_text_ratio >= 0.25:
        pdf_type = "mixed"
    elif avg_chars < 150:
        pdf_type = "low_extraction_quality"
    else:
        pdf_type = "text_based"

    # 3. Baseline font size
    if font_sizes:
        counter = collections.Counter(font_sizes)
        baseline_font_size = counter.most_common(1)[0][0]
    else:
        baseline_font_size = 10.0

    # 4. Outlines / Bookmarks
    toc = doc.get_toc() or []
    has_native_bookmarks = len(toc) > 0
    bookmark_count = len(toc)

    # 5. Repeated headers, footers, and watermarks (threshold: >= 35% of sample pages)
    min_freq = max(2, int(0.35 * sample_count))
    top_counter = collections.Counter(top_margin_lines)
    repeated_headers = [phrase for phrase, cnt in top_counter.items() if cnt >= min_freq]

    bot_counter = collections.Counter(bottom_margin_lines)
    repeated_footers = [phrase for phrase, cnt in bot_counter.items() if cnt >= min_freq]

    wm_counter = collections.Counter(full_page_blocks)
    detected_watermarks = [phrase for phrase, cnt in wm_counter.items() if cnt >= min_freq]

    # Also check if any common phrase appears repeatedly anywhere in margin lines
    for phrase, cnt in bot_counter.items():
        if cnt >= min_freq and ("government" in phrase.lower() or "copyright" in phrase.lower() or "scert" in phrase.lower() or "board" in phrase.lower()):
            if phrase not in repeated_footers:
                repeated_footers.append(phrase)

    # 6. Margin page number position
    if bottom_digits_count > top_digits_count and bottom_digits_count >= max(2, sample_count // 3):
        margin_page_number_position = "bottom"
    elif top_digits_count >= max(2, sample_count // 3):
        margin_page_number_position = "top"
    else:
        margin_page_number_position = "none"

    return DocumentProfile(
        total_pages=total_pages,
        pdf_type=pdf_type,
        baseline_font_size=baseline_font_size,
        has_native_bookmarks=has_native_bookmarks,
        bookmark_count=bookmark_count,
        is_landscape=is_landscape,
        page_width=p_width,
        page_height=p_height,
        avg_chars_per_page=avg_chars,
        image_only_pages=image_only_pages,
        repeated_headers=repeated_headers,
        repeated_footers=repeated_footers,
        detected_watermarks=detected_watermarks,
        margin_page_number_position=margin_page_number_position,
    )

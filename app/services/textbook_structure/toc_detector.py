"""Universal Table of Contents detection and multi-strategy parsing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import fitz

logger = logging.getLogger(__name__)

# TOC Heading Keywords in multiple languages
_TOC_KEYWORDS = [
    "contents",
    "table of contents",
    "index",
    "विषय-सूची",
    "विषय सूची",
    "अनुक्रमणिका",
    "पाठ्यक्रम",
    "fihrist",
]


@dataclass
class TocEntry:
    chapter_number: str
    chapter_title: str
    hierarchy_level: str  # "unit", "chapter", "reading", "appendix", "section"
    printed_start_page: int
    printed_end_page: int | None
    source_toc_page: int
    sub_readings: list[str] = field(default_factory=list)
    confidence: float = 0.90
    detection_method: str = "tabular_toc"


class UniversalTocDetector:
    """Detects and parses TOCs across single-column, multi-column, and tabular formats."""

    @classmethod
    def find_toc_pages(cls, doc: fitz.Document, max_prelim_pages: int = 35) -> list[int]:
        """Find all pages that are part of the Table of Contents."""
        toc_pages = []
        scan_limit = min(max_prelim_pages, len(doc))

        for pno in range(scan_limit):
            page = doc[pno]
            text = page.get_text() or ""
            first_lines = "\n".join(text.split("\n")[:10]).lower()
            if any(kw in first_lines for kw in _TOC_KEYWORDS):
                toc_pages.append(pno)

        # Check for multi-page TOC continuation
        if toc_pages:
            last_toc = toc_pages[-1]
            if last_toc + 1 < scan_limit:
                next_page = doc[last_toc + 1]
                next_text = next_page.get_text() or ""
                # If next page has page ranges (e.g. '123-145') or chapter lines without a new major heading
                if re.search(r"\b\d+\s*[-–—]\s*\d+\b", next_text) or re.search(r"^\s*(?:chapter|unit|\d+\.)\s+", next_text, re.I | re.M):
                    if not any(kw in next_text[:300].lower() for kw in ["preface", "acknowledgement", "foreword"]):
                        toc_pages.append(last_toc + 1)

        return toc_pages

    @classmethod
    def parse_toc(cls, doc: fitz.Document, toc_pages: list[int] | None = None) -> list[TocEntry]:
        """Parse identified TOC pages using multi-strategy layout and regex parsing."""
        if toc_pages is None:
            toc_pages = cls.find_toc_pages(doc)

        if not toc_pages:
            return []

        entries: list[TocEntry] = []
        for pno in toc_pages:
            page = doc[pno]
            # Try tabular extraction first
            tabular_entries = cls._parse_tabular_toc(page, pno)
            if tabular_entries:
                entries.extend(tabular_entries)
            else:
                # Fall back to line-based / dotted leader parsing
                line_entries = cls._parse_line_toc(page, pno)
                entries.extend(line_entries)

        return entries

    @classmethod
    def _parse_tabular_toc(cls, page: fitz.Page, pno: int) -> list[TocEntry]:
        """Parse multi-column / tabular TOCs using block layout and word coordinate clustering."""
        blocks = page.get_text("blocks")
        words = page.get_text("words")
        w_page = page.rect.width
        h_page = page.rect.height

        entries: list[TocEntry] = []

        for b in blocks:
            txt = b[4].strip()
            lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
            if not lines:
                continue

            # Check if block has a unit or chapter number at start
            first_line = lines[0]
            unit_match = re.match(r"^(?:(?:Unit|Chapter|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+)[\.\s]*$", first_line, re.I)

            # Check if block has page range (e.g. '1-14', '15-30', '123-141')
            page_range_match = None
            for ln in lines:
                m_r = re.match(r"^(\d+)\s*[-–—to]+\s*(\d+)$", ln, re.I)
                if m_r:
                    page_range_match = m_r
                    break

            # Check if block represents an Appendix row (e.g. 'Appendices // 142-152' or 'Listening Texts // 142-146')
            is_appendix = any("appendix" in ln.lower() or "appendices" in ln.lower() or "listening texts" in ln.lower() for ln in lines)

            if unit_match and page_range_match:
                ch_num = unit_match.group(1)
                start_p = int(page_range_match.group(1))
                end_p = int(page_range_match.group(2))

                # Extract words within the vertical bounds of this block
                y_min, y_max = b[1] - 2, b[3] + 2
                row_words = [w for w in words if y_min <= w[1] <= y_max]

                # Adaptive column extraction based on relative page width
                # Col 0 (Unit/Num): x < 0.12 * W
                # Col 1 (Theme/Title): 0.12 * W <= x < 0.31 * W
                # Col 2 (Readings/Subtopics): 0.31 * W <= x < 0.72 * W
                theme_words = [w for w in row_words if 0.12 * w_page <= w[0] < 0.31 * w_page]
                theme = " ".join(w[4] for w in sorted(theme_words, key=lambda x: (x[1], x[0]))).strip()

                content_words = [w for w in row_words if 0.31 * w_page <= w[0] < 0.72 * w_page]
                content_txt = " ".join(w[4] for w in sorted(content_words, key=lambda x: (x[1], x[0]))).strip()

                # Extract sub-readings if present (e.g. A. The Tattered Blanket, B. My Mother)
                sub_readings = []
                if content_txt:
                    sub_parts = re.split(r"(?=[A-C]\.\s+)", content_txt)
                    for sp in sub_parts:
                        clean_sp = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)(?:-[A-Za-z]+)?\b", "", sp, flags=re.I).strip()
                        if clean_sp and len(clean_sp) > 3:
                            sub_readings.append(clean_sp)

                title = f"Unit {ch_num}: {theme}" if theme else f"Unit {ch_num}"
                entries.append(
                    TocEntry(
                        chapter_number=ch_num,
                        chapter_title=title,
                        hierarchy_level="unit",
                        printed_start_page=start_p,
                        printed_end_page=end_p,
                        source_toc_page=pno,
                        sub_readings=sub_readings,
                        confidence=0.96,
                        detection_method="tabular_toc",
                    )
                )

            elif is_appendix and page_range_match:
                start_p = int(page_range_match.group(1))
                end_p = int(page_range_match.group(2))
                app_title = lines[0]
                entries.append(
                    TocEntry(
                        chapter_number="App",
                        chapter_title=app_title,
                        hierarchy_level="appendix",
                        printed_start_page=start_p,
                        printed_end_page=end_p,
                        source_toc_page=pno,
                        confidence=0.92,
                        detection_method="tabular_toc",
                    )
                )

        return entries

    @classmethod
    def _parse_line_toc(cls, page: fitz.Page, pno: int) -> list[TocEntry]:
        """Parse line-based or dotted-leader TOC entries."""
        lines = [ln.strip() for ln in page.get_text().split("\n") if ln.strip()]
        entries: list[TocEntry] = []

        line_pattern = re.compile(
            r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*?)\s+[\.\s_]*(\d+)(?:\s*[-–—to]+\s*(\d+))?$",
            re.IGNORECASE,
        )
        dotted_pattern = re.compile(
            r"^(.*?)\s+[\.·\s_]{3,}\s*(\d+)(?:\s*[-–—to]+\s*(\d+))?$",
            re.IGNORECASE,
        )

        for line in lines:
            m = line_pattern.match(line)
            if m:
                ch_num = m.group(1)
                title = m.group(2).strip()
                sp = int(m.group(3))
                ep = int(m.group(4)) if m.group(4) else None
                entries.append(
                    TocEntry(
                        chapter_number=ch_num,
                        chapter_title=title,
                        hierarchy_level="chapter",
                        printed_start_page=sp,
                        printed_end_page=ep,
                        source_toc_page=pno,
                        confidence=0.88,
                        detection_method="line_toc",
                    )
                )
                continue

            m2 = dotted_pattern.match(line)
            if m2:
                title_raw = m2.group(1).strip()
                sp = int(m2.group(2))
                ep = int(m2.group(3)) if m2.group(3) else None

                # Extract chapter number if present in title
                ch_m = re.match(r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*)$", title_raw, re.I)
                if ch_m:
                    ch_num = ch_m.group(1)
                    title = ch_m.group(2).strip()
                else:
                    ch_num = str(len(entries) + 1)
                    title = title_raw

                entries.append(
                    TocEntry(
                        chapter_number=ch_num,
                        chapter_title=title,
                        hierarchy_level="chapter",
                        printed_start_page=sp,
                        printed_end_page=ep,
                        source_toc_page=pno,
                        confidence=0.85,
                        detection_method="dotted_toc",
                    )
                )

        return entries

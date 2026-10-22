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


_MONTH_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)[.,]?$",
    re.IGNORECASE,
)

# Upper-case diagonal watermark words; mixed-case 'Telangana' in a title must survive.
_WATERMARK_RE = re.compile(r"^(SCERT|TELANGANA)[,.]?$")


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
            # Reading order: raw extraction order can put the heading after the table body.
            lines = [ln for ln in (page.get_text(sort=True) or "").split("\n") if ln.strip()]
            first_lines = "\n".join(lines[:10]).lower()
            if any(kw in first_lines for kw in _TOC_KEYWORDS):
                toc_pages.append(pno)
        # Later hits are body pages that mention 'contents' (e.g. an exercise on reading a TOC).
        toc_pages = [p for p in toc_pages if p - toc_pages[0] <= 2]

        if not toc_pages:
            # Contents tables without any heading: accept a page that reads as one.
            toc_pages = [pno for pno in range(scan_limit) if len(cls._parse_row_toc(doc[pno], pno)) >= 5]

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

        # Layouts differ per textbook, so every parser reads the contents pages and the
        # result that best looks like a contents table (ordered pages, ranges) wins.
        parsers = (cls._parse_reading_rows_toc, cls._parse_tabular_toc, cls._parse_line_toc, cls._parse_row_toc)
        results = [[e for pno in toc_pages for e in parser(doc[pno], pno)] for parser in parsers]
        return max(results, key=cls._toc_score)  # ties keep the earlier parser

    @staticmethod
    def _toc_score(entries: list[TocEntry]) -> int:
        """Chapters/units whose start pages increase down the list, plus those with an end page."""
        score, last = 0, -1
        for e in entries:
            if e.hierarchy_level not in ("chapter", "unit"):
                continue
            if e.printed_start_page > last:
                score += 1
                last = e.printed_start_page
            if e.printed_end_page:
                score += 1
        return score

    @staticmethod
    def _visual_rows(page: fitz.Page) -> list[list[tuple]]:
        """Words regrouped into visual rows (left to right); table cells often extract one per line."""
        rows: list[list[tuple]] = []
        for w in sorted(page.get_text("words"), key=lambda w: (w[3], w[0])):
            if rows and abs(rows[-1][0][3] - w[3]) <= 3:
                rows[-1].append(w)
            else:
                rows.append([w])
        return [sorted(r, key=lambda w: w[0]) for r in rows]

    @classmethod
    def _parse_reading_rows_toc(cls, page: fitz.Page, pno: int) -> list[TocEntry]:
        """Unit tables labelled by reading: '1. A. Reading <title>' / 'B. Reading (Poem) <title> <month> 4-18'.

        Each numbered 'A' row starts a unit; B/C/... rows are its readings; the unit's page
        range sits on any of its rows. Rows with a range before the first unit (e.g. a
        readiness programme) become chapters, 'Appendix' rows become appendices.
        """
        units: list[dict[str, Any]] = []
        extras: list[TocEntry] = []
        pre_rows: list[tuple[list[tuple], tuple[int, int]]] = []
        title_x: float | None = None

        for row in cls._visual_rows(page):
            words = [w for w in row if not _WATERMARK_RE.match(w[4])]
            tokens = [w[4] for w in words]
            i = 0
            num = None
            if tokens and re.fullmatch(r"\d{1,2}\.?", tokens[0]):
                num, i = tokens[0].rstrip("."), 1
            letter = poem = None
            if i + 1 < len(tokens) and re.fullmatch(r"[A-E]\.?", tokens[i]) and re.fullmatch(r"Re\w*ding", tokens[i + 1], re.I):
                letter, i = tokens[i][0], i + 2
                if i < len(tokens) and tokens[i].lower() == "(poem)":
                    poem, i = True, i + 1

            rng = None
            title: list[str] = []
            for t in tokens[i:]:
                m = re.fullmatch(r"(\d+)[-–—](\d+)", t)
                if m:
                    rng = (int(m.group(1)), int(m.group(2)))
                elif not all(_MONTH_RE.match(p) for p in re.split(r"[/,\-–]", t) if p):
                    title.append(t)

            if letter and num and letter == "A":
                if title_x is None and i < len(words):
                    title_x = words[i][0]
                units.append({"num": num, "readings": [], "range": None})
            elif not letter:
                if tokens and tokens[0].lower().startswith("appendi") and rng:
                    extras.append(TocEntry("App", " ".join(title), "appendix", rng[0], rng[1], pno, confidence=0.92, detection_method="reading_rows_toc"))
                elif not units and rng:
                    pre_rows.append((words, rng))
                continue
            if not units or not title:
                continue
            units[-1]["readings"].append(f"{letter}. {' '.join(title)}" + (" (Poem)" if poem else ""))
            if rng:
                units[-1]["range"] = rng

        units = [u for u in units if u["range"]]
        if len(units) < 2:
            return []

        entries = []
        for words, (sp, ep) in pre_rows:
            title = " ".join(w[4] for w in words if w[2] > (title_x or 0) and not re.search(r"\d", w[4]) and not _MONTH_RE.match(w[4]))
            if title:
                entries.append(TocEntry("0", title, "chapter", sp, ep, pno, confidence=0.9, detection_method="reading_rows_toc"))
        for u in units:
            first = re.sub(r"^[A-E]\.\s*|\s*\(Poem\)$", "", u["readings"][0])
            entries.append(TocEntry(
                chapter_number=u["num"],
                chapter_title=f"Unit {u['num']}: {first}",
                hierarchy_level="unit",
                printed_start_page=u["range"][0],
                printed_end_page=u["range"][1],
                source_toc_page=pno,
                sub_readings=u["readings"],
                confidence=0.95,
                detection_method="reading_rows_toc",
            ))
        return entries + extras

    @classmethod
    def _parse_row_toc(cls, page: fitz.Page, pno: int) -> list[TocEntry]:
        """Table TOCs whose cells extract one per line (e.g. 'No. | Chapter | Page | Periods | Month').

        Words are regrouped into visual rows; a row is '<n>. <title words> <numeric cells...>'.
        The title stops at the first numeric or month cell. Column order varies between books,
        so the page column is whichever numeric position (first or last) increases down the table.
        """
        parsed: list[tuple[str, str, list[int]]] = []
        for row in cls._visual_rows(page):
            tokens = [w[4] for w in row]
            num = re.fullmatch(r"(\d{1,2})\.?", tokens[0])
            if not num or len(tokens) < 3:
                continue
            title_words: list[str] = []
            numbers: list[int] = []
            for t in tokens[1:]:
                if re.fullmatch(r"\d+(?:[-–—]\d+)?", t):  # '12', '1-16'
                    numbers.extend(int(x) for x in re.findall(r"\d+", t))
                elif numbers or (title_words and _MONTH_RE.match(t)):
                    continue  # month / trailing cells
                else:
                    title_words.append(t)
            if title_words and numbers:
                parsed.append((num.group(1), " ".join(title_words), numbers))

        if len(parsed) < 2:
            return []
        for pos in (0, -1):
            pages = [nums[pos] for _, _, nums in parsed]
            # A real TOC lists pages in order; anything else is a stray numbered list or periods.
            if all(a < b for a, b in zip(pages, pages[1:])):
                return [
                    TocEntry(
                        chapter_number=n,
                        chapter_title=title,
                        hierarchy_level="chapter",
                        printed_start_page=pg,
                        printed_end_page=None,
                        source_toc_page=pno,
                        confidence=0.88,
                        detection_method="row_toc",
                    )
                    for (n, title, _), pg in zip(parsed, pages)
                ]
        return []

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

                theme = re.sub(rf"^{re.escape(ch_num)}[.)]?\s*", "", theme)
                sub_readings = []
                if re.search(r"(?:^|\s)[A-C]\.\s+\S", content_txt):
                    # Real readings are listed with markers (e.g. A. The Tattered Blanket, B. My Mother)
                    for sp in re.split(r"(?=(?:^|\s)[A-C]\.\s+)", content_txt):
                        clean_sp = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)(?:-[A-Za-z]+)?\b", "", sp, flags=re.I).strip()
                        if clean_sp and len(clean_sp) > 3:
                            sub_readings.append(clean_sp)
                else:
                    # No markers: this column is just the title running long (plus the month column).
                    tail = [t for t in content_txt.split() if not all(_MONTH_RE.match(p) for p in re.split(r"[/,\-]", t) if p)]
                    theme = " ".join([theme, *tail]).strip()

                if sub_readings:
                    title = f"Unit {ch_num}: {theme}" if theme else f"Unit {ch_num}"
                else:
                    title = theme or f"Chapter {ch_num}"
                entries.append(
                    TocEntry(
                        chapter_number=ch_num,
                        chapter_title=title,
                        hierarchy_level="unit" if sub_readings else "chapter",
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

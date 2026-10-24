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

_PAGE_CELL_RE = re.compile(r"(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?")


BACK_MATTER_RE = re.compile(
    r"^(appendi|annexure|answers?\b|glossary|bibliograph|references\b|about the textbook|(listening )?texts? for listening|listening texts)",
    re.I,
)


def _is_month(token: str) -> bool:
    parts = [p for p in re.split(r"[/,\-–&]", token) if p]
    return bool(parts) and all(_MONTH_RE.match(p) for p in parts)


def clean_title(title: str) -> tuple[str | None, str]:
    """Split a leading chapter number off a TOC title and drop table cells that leaked into it.

    'Our Earth 1-13' / 'Real Numbers 15' / 'Sound Revision' -> title only;
    '11A. National Movement' -> ('11A', 'National Movement'); '11August/September Diversity' -> 'Diversity'.
    """
    words = [w for w in title.split() if not (w[:1].isdigit() and _is_month(w.lstrip("0123456789")))]
    while words and (re.fullmatch(r"\d{1,3}(?:[-–—]\d{1,3})?[.,]?", words[-1]) or _is_month(words[-1]) or words[-1].lower() == "revision"):
        words.pop()
    text = " ".join(words)
    m = re.match(r"(\d{1,2}[A-Z]?)[.)]\s+(?=\S)", text)
    num = m.group(1).lstrip("0") or "0" if m else None
    return num, (text[m.end():] if m else text).strip(" -–:,") or title.strip()


def _split_glued_readings(readings: list[str]) -> list[str]:
    """'B. My Mother (Poem) C.A Letter' -> two readings. Only the next letter splits, so 'A. C.V.Raman' stays whole."""
    out: list[str] = []
    for r in readings:
        while r[:1] in "ABCD":
            nxt = chr(ord(r[0]) + 1)
            m = re.search(rf"\s{nxt}\.\s*(?:Re\w*ding\s+)?", r)
            if not m:
                break
            out.append(r[: m.start()])
            r = f"{nxt}. {r[m.end():]}"
        out.append(r)
    return out


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
        parsers = (cls._parse_reading_rows_toc, cls._parse_tabular_toc, cls._parse_line_toc, cls._parse_row_toc, cls._parse_record_toc)
        results = []
        for parser in parsers:
            entries: list[TocEntry] = []
            for pno in toc_pages:
                page_entries = parser(doc[pno], pno)
                # A 'continuation' page whose pages restart is body text (a chapter's first table), not TOC.
                if entries and page_entries and page_entries[0].printed_start_page <= entries[-1].printed_start_page:
                    continue
                entries.extend(page_entries)
            results.append(entries)
        best = max(results, key=cls._toc_score)  # ties keep the earlier parser
        for e in best:
            num, e.chapter_title = clean_title(e.chapter_title)
            if num and e.hierarchy_level == "chapter":
                e.chapter_number = num
            elif e.chapter_number.isdigit():
                e.chapter_number = e.chapter_number.lstrip("0") or "0"
            if BACK_MATTER_RE.match(e.chapter_title):
                e.hierarchy_level = "appendix"
        return best

    @staticmethod
    def _toc_score(entries: list[TocEntry]) -> int:
        """Chapters/units whose start pages increase down the list, plus those with an end page or readings."""
        score, last = 0, -1
        for e in entries:
            if e.hierarchy_level not in ("chapter", "unit"):
                continue
            if e.printed_start_page > last:
                score += 1
                last = e.printed_start_page
            if e.printed_end_page:
                score += 1
            if e.sub_readings:
                score += 1
        return score

    @staticmethod
    def _page_words(page: fitz.Page) -> list[tuple]:
        """Words minus rotated text: diagonal watermarks extract as stray letters glued into titles."""
        rotated = [
            (fitz.Rect(c["bbox"]), "".join(x["c"] for x in s["chars"]))
            for b in page.get_text("rawdict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for ln in b.get("lines", []) if abs(ln["dir"][1]) > 0.05
            for s in ln["spans"] for c in s["chars"]
        ]
        words = page.get_text("words")
        if not rotated:
            return words
        # Rotated glyph boxes are large, so a word must also spell part of the rotated text.
        return [
            w for w in words
            if not any(w[4] in text and r.contains(fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)) for r, text in rotated)
        ]

    @classmethod
    def _visual_rows(cls, page: fitz.Page) -> list[list[tuple]]:
        """Words regrouped into visual rows (left to right); table cells often extract one per line."""
        rows: list[list[tuple]] = []
        for w in sorted(cls._page_words(page), key=lambda w: (w[3], w[0])):
            if rows and abs(rows[-1][0][3] - w[3]) <= 3:
                rows[-1].append(w)
            else:
                rows.append([w])
        return [sorted(r, key=lambda w: w[0]) for r in rows]

    @classmethod
    def _cells(cls, page: fitz.Page) -> list[tuple]:
        """Visual-row words with split page ranges ('1 - 25', '291-' '306') merged into one cell."""
        out: list[tuple] = []
        for row in cls._visual_rows(page):
            merged: list[tuple] = []
            for w in row:
                glued = re.fullmatch(r"([A-Za-z]{3,})(\d{1,3})", w[4])  # 'Annexure64'
                if glued:
                    cut = w[0] + (w[2] - w[0]) * len(glued.group(1)) / len(w[4])
                    merged.append((w[0], w[1], cut, w[3], glued.group(1)))
                    w = (cut, w[1], w[2], w[3], glued.group(2), "glued")
                prev = merged[-1] if merged else None
                if prev and w[0] - prev[2] < 12 and (
                    (re.fullmatch(r"\d{1,3}\s*[-–—]?", prev[4]) and re.fullmatch(r"[-–—]\s*\d{0,3}|\d{1,3}", w[4]) and not re.fullmatch(r"\d+", prev[4] + w[4]))
                    or re.fullmatch(r"\d{1,3}\s*[-–—]", prev[4]) and re.fullmatch(r"\d{1,3}", w[4])
                ):
                    merged[-1] = (prev[0], min(prev[1], w[1]), w[2], max(prev[3], w[3]), prev[4] + w[4])
                else:
                    merged.append(w)
            out.extend(merged)
        return out

    @staticmethod
    def _columns(cells: list[tuple]) -> list[list[tuple]]:
        """Cells grouped into columns by overlapping x-extent, each column sorted top to bottom."""
        cols: list[list[tuple]] = []
        spans: list[list[float]] = []
        for c in sorted(cells, key=lambda c: c[0]):
            if spans and c[0] <= spans[-1][1]:
                cols[-1].append(c)
                spans[-1][1] = max(spans[-1][1], c[2])
            else:
                cols.append([c])
                spans.append([c[0], c[2]])
        return [sorted(col, key=lambda c: c[1]) for col in cols]

    @staticmethod
    def _increasing(col: list[tuple], key) -> list[tuple]:
        """Longest top-to-bottom run of strictly increasing values (drops stray cells)."""
        best: list[list[tuple]] = []
        for c in col:
            v = key(c)
            fit = [b for b in best if key(b[-1]) < v]
            best.append(max(fit, key=len) + [c] if fit else [c])
        return max(best, key=len, default=[])

    @staticmethod
    def _align(nums: list[tuple], pages: list[tuple], y) -> dict[int, tuple]:
        """Order-preserving match of number cells to page cells (closest in y overall); pages may go unnumbered."""
        n, m = len(nums), len(pages)
        inf = float("inf")
        cost = [[inf] * (m + 1) for _ in range(n + 1)]
        cost[0] = [0.0] * (m + 1)
        for i in range(1, n + 1):
            for j in range(i, m + 1):
                cost[i][j] = min(cost[i][j - 1], cost[i - 1][j - 1] + abs(y(nums[i - 1]) - y(pages[j - 1])))
        out: dict[int, tuple] = {}
        i, j = n, m
        while i > 0:
            if cost[i][j] == cost[i][j - 1] and j > i:
                j -= 1
            else:
                out[j - 1] = nums[i - 1]
                i, j = i - 1, j - 1
        return out

    @classmethod
    def _parse_record_toc(cls, page: fitz.Page, pno: int) -> list[TocEntry]:
        """Ruled table TOCs ('No. | Name | Periods | Month | Page') whose cells drift across visual rows.

        The page column is the numeric column whose values increase down the page and run highest;
        the number column is the leftmost column of increasing small integers. Each page cell anchors
        one record; every other word joins the record whose anchors are vertically nearest, so a title
        printed above, beside or below its numbers still lands in the right row.
        """
        cells = cls._cells(page)
        numeric = [c for c in cells if _PAGE_CELL_RE.fullmatch(c[4].rstrip("."))]
        start = lambda c: int(_PAGE_CELL_RE.fullmatch(c[4].rstrip(".")).group(1))
        runs = [r for r in (cls._increasing(col, start) for col in cls._columns(numeric)) if len(r) >= 3]
        if not runs:
            return []
        is_range = lambda c: bool(_PAGE_CELL_RE.fullmatch(c[4].rstrip(".")).group(2))
        pages = max(runs, key=lambda r: (any(map(is_range, r)), start(r[-1])))
        # A page shifted out of its column (a short row printed left-aligned, 'Annexure64') still fits the order.
        for c in numeric:
            if c not in pages and ((is_range(c) and c[2] > pages[0][0] - 150) or c[5:6] == ("glued",)):
                before = [p for p in pages if p[1] < c[1]]
                after = [p for p in pages if p[1] > c[1]]
                if (not before or start(before[-1]) < start(c)) and (not after or start(c) < start(after[0])):
                    pages.insert(len(before), c)
        title_x1 = sorted(p[0] for p in pages)[len(pages) // 2]  # median: rescued cells sit left of the column
        nums = [r for r in runs if r is not pages and r[0][2] < title_x1 and start(r[-1]) <= 60 and len(r) <= len(pages)]
        nums = min(nums, key=lambda r: r[0][0], default=[])
        num_x1 = max((c[2] for c in nums), default=0)
        # Labels such as '11A.' share the number column without being plain integers.
        nums = sorted(
            nums + [c for c in cells if c not in nums and c[2] <= num_x1 + 2 and re.fullmatch(r"\d{1,2}[A-Za-z]\.?", c[4])],
            key=lambda c: c[1],
        )[: len(pages)]
        used = {id(c) for c in pages + nums}

        def mid(c: tuple) -> float:
            return (c[1] + c[3]) / 2

        number_of = cls._align(nums, pages, mid)
        centers = [(mid(p) + mid(number_of[k])) / 2 if k in number_of else mid(p) for k, p in enumerate(pages)]
        gap = sorted(b - a for a, b in zip(centers, centers[1:]))[len(centers) // 2 - 1] if len(centers) > 1 else 30
        month_x0 = sorted(c[0] for c in cells if _is_month(c[4]))
        # Anything in the month column is not title, misspelt months included ('Ferbruary').
        title_x1 = min(title_x1, month_x0[len(month_x0) // 2] - 2) if len(month_x0) >= 3 else title_x1
        # A month away from every record is a schedule row ('Revision | March'), not a chapter.
        schedule_ys = [c[3] for c in cells if _is_month(c[4]) and min(abs(mid(c) - y) for y in centers) > gap * 0.75]
        records: list[list[tuple]] = [[] for _ in pages]
        for c in cells:
            if id(c) in used or c[2] > title_x1 + 2 or any(abs(c[3] - y) <= 3 for y in schedule_ys):
                continue
            if _is_month(c[4]) or re.fullmatch(r"\d{1,2}\.?", c[4]):  # month, periods, second number column
                continue
            i = min(range(len(centers)), key=lambda k: abs(centers[k] - mid(c)))
            # Tight above (keeps the header row out); a full row below the last record (its wrapped title).
            if abs(centers[i] - mid(c)) <= gap * (1.05 if i == len(centers) - 1 and mid(c) > centers[i] else 0.75):
                records[i].append(c)

        entries = []
        prev_num, part = "0", 0
        for k, (p, words) in enumerate(zip(pages, records)):
            m = _PAGE_CELL_RE.fullmatch(p[4].rstrip("."))
            title = " ".join(w[4] for w in sorted(words, key=lambda w: (round(w[1] / 4), w[0]))).strip()
            if not title:
                continue
            if not nums:
                num = str(k + 1)
            elif k in number_of:
                num, prev_num, part = number_of[k][4].rstrip("."), number_of[k][4].rstrip("."), 0
            else:  # unnumbered row in a numbered table: a further part of the previous chapter
                part += 1
                num = f"{prev_num}{chr(ord('A') + part)}"
            entries.append(TocEntry(
                chapter_number=num,
                chapter_title=title,
                hierarchy_level="chapter",
                printed_start_page=int(m.group(1)),
                printed_end_page=int(m.group(2)) if m.group(2) else None,
                source_toc_page=pno,
                confidence=0.9,
                detection_method="record_toc",
            ))
        return entries if len(entries) >= 3 else []

    @classmethod
    def _parse_reading_rows_toc(cls, page: fitz.Page, pno: int) -> list[TocEntry]:
        """Unit tables whose readings are lettered A/B/C, one reading per row.

        Handles e.g. '1. A. Reading <title>' / 'B. Reading (Poem) <title> <month> 4-18' (no theme)
        and '1. Personality | A. <title>' / 'Development | B. <title> June 1-32' (theme column).
        A numbered row holding the 'A' label starts a unit; later lettered rows are its readings;
        unlabelled rows continue the theme (left of the labels) or the last reading (wrapped title).
        The unit's page range sits on any of its rows. Ranged rows before the first unit
        (e.g. a readiness programme) become chapters; ranged rows under 'Appendices' become appendices.
        """
        units: list[dict[str, Any]] = []
        extras: list[TocEntry] = []
        pre_rows: list[tuple[list[tuple], tuple[int, int]]] = []
        title_x: float | None = None
        label_x: float | None = None
        in_appendix = False

        for row in cls._visual_rows(page):
            words = row
            tokens = [w[4] for w in words]
            if not tokens:
                continue
            # Reading label: 'B.' or 'B Reading'; only at the row start or right after the unit number + theme.
            li = next(
                (k for k, t in enumerate(tokens)
                 if re.fullmatch(r"[A-E]\.", t)
                 or (re.fullmatch(r"[A-E]", t) and k + 1 < len(tokens) and re.fullmatch(r"Re\w*ding", tokens[k + 1], re.I))),
                None,
            )
            theme_tail: list[str] = []
            if li and not re.fullmatch(r"\d{1,2}\.?", tokens[0]):
                # A wrapped theme line sharing the row with a later label ('Technology  B. Preteen ...').
                if units and label_x and all(w[2] < label_x for w in words[:li]):
                    theme_tail = tokens[:li]
                else:
                    li = None
            i = 0 if li is None else li + 1
            poem = False
            if li is not None:
                if i < len(tokens) and re.fullmatch(r"Re\w*ding", tokens[i], re.I):
                    i += 1
                if i < len(tokens) and tokens[i].lower() == "(poem)":
                    poem, i = True, i + 1

            rng = None
            title: list[str] = []
            for t in tokens[i:]:
                m = re.fullmatch(r"(\d+)[-–—](\d+)", t)
                if m and int(m.group(1)) < int(m.group(2)):  # '2026-27' is a year, not pages
                    rng = (int(m.group(1)), int(m.group(2)))
                elif not all(_MONTH_RE.match(p) for p in re.split(r"[/,\-–]", t) if p):
                    title.append(t)

            if li is None:
                if tokens[0].lower().startswith("appendi"):
                    in_appendix = True
                if in_appendix:
                    if rng and not re.fullmatch(r"appendices", " ".join(title), re.I):  # group header
                        extras.append(TocEntry("App", " ".join(title), "appendix", rng[0], rng[1], pno, confidence=0.92, detection_method="reading_rows_toc"))
                elif not units:
                    if rng:
                        pre_rows.append((words, rng))
                else:
                    if rng:
                        units[-1]["range"] = rng
                    if label_x and all(w[2] < label_x for w in words):
                        if units[-1]["theme"]:
                            units[-1]["theme"] += " " + " ".join(title)
                    elif title and units[-1]["readings"]:
                        units[-1]["readings"][-1] += " " + " ".join(title)
                continue

            letter = tokens[li][0]
            if theme_tail:
                units[-1]["theme"] += " " + " ".join(theme_tail)
            elif li and letter == "A":
                label_x = label_x or words[li][0]
                if title_x is None and i < len(words):
                    title_x = words[i][0]
                units.append({"num": tokens[0].rstrip("."), "theme": " ".join(tokens[1:li]), "readings": [], "range": None})
            elif li:
                continue  # numbered row with a non-A label: not this layout
            if not units:
                continue
            if title:
                units[-1]["readings"].append(f"{letter}. {' '.join(title)}" + (" (Poem)" if poem else ""))
            if rng:
                units[-1]["range"] = rng

        units = [u for u in units if u["range"] and u["readings"]]
        for u in units:
            u["readings"] = _split_glued_readings(u["readings"])
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
                chapter_title=f"Unit {u['num']}: {u['theme'] or first}",
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

"""Adaptive body heading detection with typographic prominence and layout analysis."""

from __future__ import annotations

import logging
import re
from typing import Any

import fitz

from app.services.textbook_structure.profiler import DocumentProfile

logger = logging.getLogger(__name__)

# Word to number mapping
_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def parse_chapter_number_str(raw: str) -> str:
    cleaned = raw.strip().rstrip(".:-")
    if cleaned.isdigit():
        return str(int(cleaned))
    from app.services.textbook_structure.offset_calibrator import roman_to_int
    as_roman = roman_to_int(cleaned)
    if as_roman is not None:
        return str(as_roman)
    as_word = _WORD_NUMS.get(cleaned.lower())
    if as_word is not None:
        return str(as_word)
    return cleaned


class AdaptiveHeadingDetector:
    """Scans body pages for prominent chapter, unit, and section headings."""

    HEADING_REGEX = re.compile(
        r"^(?:CHAPTER|Chapter|UNIT|Unit|LESSON|Lesson|अध्याय)\s*(\d+|[IVXLCDM]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)?(?:\s*[:\.\-–—\s]+(.*))?$",
        re.IGNORECASE,
    )
    READING_REGEX = re.compile(
        r"^(?:Reading\s+([A-C]))\b",
        re.IGNORECASE,
    )
    NEGATIVE_REGEX = re.compile(
        r"^(?:fig(?:ure)?\s*\d+|table\s*\d+|diagram\s*\d+|exercise|exercises|practice|questions|review|in\s+chapter|refer\s+to|as\s+seen\s+in|source:|oral\s+discourse)",
        re.IGNORECASE,
    )

    @classmethod
    def scan_headings(cls, doc: fitz.Document, profile: DocumentProfile) -> list[dict[str, Any]]:
        candidates = []
        base_sz = profile.baseline_font_size or 10.0
        total_pages = len(doc)

        for pno in range(total_pages):
            page = doc[pno]
            h = page.rect.height
            d = page.get_text("dict")
            blocks = d.get("blocks", [])

            for b_idx, b in enumerate(blocks):
                if "lines" not in b:
                    continue
                for line in b["lines"]:
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        sz = span.get("size", 0.0)
                        flags = span.get("flags", 0)  # 2 = bold
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        y0 = bbox[1]

                        # Must be in upper 65% of page and not in extreme header
                        if y0 < 15 or y0 > 0.65 * h:
                            continue

                        # Reject negative patterns
                        if cls.NEGATIVE_REGEX.search(txt):
                            continue

                        # Case A: Explicit Chapter/Unit/अध्याय pattern match
                        m = cls.HEADING_REGEX.match(txt)
                        if m:
                            ch_num_raw = m.group(1)
                            remainder = (m.group(2) or "").strip()

                            # If number was not on the same span (e.g. standalone "CHAPTER" or "अध्याय")
                            if not ch_num_raw:
                                # Look for number in adjacent spans or nearby blocks
                                for nearby_b in blocks[max(0, b_idx - 1): min(len(blocks), b_idx + 4)]:
                                    for nl in nearby_b.get("lines", []):
                                        for ns in nl.get("spans", []):
                                            ntxt = ns.get("text", "").strip()
                                            if ntxt.isdigit() and int(ntxt) < 100:
                                                ch_num_raw = ntxt
                                                break
                                        if ch_num_raw:
                                            break
                                    if ch_num_raw:
                                        break

                            if not ch_num_raw:
                                continue

                            title = remainder
                            if not title:
                                # Look at next lines or nearby prominent spans
                                title_spans = []
                                for subsequent_span in line.get("spans", []):
                                    sub_txt = subsequent_span.get("text", "").strip()
                                    if sub_txt and sub_txt != txt and sub_txt != ch_num_raw:
                                        title_spans.append(sub_txt)
                                if not title_spans:
                                    for nearby_b in blocks[max(0, b_idx - 1): min(len(blocks), b_idx + 4)]:
                                        for nl in nearby_b.get("lines", []):
                                            for ns in nl.get("spans", []):
                                                st = ns.get("text", "").strip()
                                                if (
                                                    st
                                                    and len(st) > 2
                                                    and st != txt
                                                    and st != ch_num_raw
                                                    and not cls.NEGATIVE_REGEX.search(st)
                                                    and ns.get("size", 0.0) >= 1.2 * base_sz
                                                ):
                                                    title_spans.append(st)
                                title = " ".join(title_spans[:2])

                            candidates.append({
                                "chapter_number": parse_chapter_number_str(ch_num_raw),
                                "chapter_title": title or f"Chapter {ch_num_raw}",
                                "physical_page": pno,
                                "font_size": sz,
                                "is_bold": bool(flags & 2),
                                "confidence": 0.88,
                                "detection_method": "layout_heading",
                                "hierarchy_level": "chapter",
                            })
                            break

                        # Case B: Language textbook Reading A/B/C match
                        m_read = cls.READING_REGEX.match(txt)
                        if m_read:
                            reading_letter = m_read.group(1).upper()
                            # Title usually follows on next line or remainder
                            remainder = txt[m_read.end():].lstrip(" :.-")
                            title = remainder
                            if not title and b_idx + 1 < len(blocks):
                                next_b = blocks[b_idx + 1]
                                next_txts = [
                                    ns.get("text", "").strip()
                                    for nl in next_b.get("lines", [])
                                    for ns in nl.get("spans", [])
                                    if ns.get("text", "").strip()
                                ]
                                title = " ".join(next_txts[:2])

                            candidates.append({
                                "chapter_number": reading_letter,
                                "chapter_title": f"Reading {reading_letter}: {title}" if title else f"Reading {reading_letter}",
                                "physical_page": pno,
                                "font_size": sz,
                                "is_bold": bool(flags & 2),
                                "confidence": 0.82,
                                "detection_method": "reading_heading",
                                "hierarchy_level": "reading",
                            })
                            break

                        # Case C: Prominent standalone number heading (sz >= 1.6 * base)
                        if (
                            sz >= 1.6 * base_sz
                            and (bool(flags & 2) or sz >= 2.0 * base_sz)
                            and (txt.isdigit() or parse_chapter_number_str(txt).isdigit())
                            and len(txt) <= 4
                        ):
                            ch_num = parse_chapter_number_str(txt)
                            # Find title in nearby blocks with large font
                            title_candidates = []
                            for ob in blocks:
                                for ol in ob.get("lines", []):
                                    for os_span in ol.get("spans", []):
                                        otxt = os_span.get("text", "").strip()
                                        osz = os_span.get("size", 0.0)
                                        if (
                                            osz >= 1.3 * base_sz
                                            and not otxt.isdigit()
                                            and otxt.upper() not in ("CHAPTER", "UNIT", "LESSON", "अध्याय")
                                            and osz < 65
                                            and not cls.NEGATIVE_REGEX.search(otxt)
                                        ):
                                            title_candidates.append((osz, otxt))
                            if title_candidates:
                                max_sz = max(s[0] for s in title_candidates)
                                title = " ".join(s[1] for s in title_candidates if abs(s[0] - max_sz) <= 2.5)
                            else:
                                title = f"Chapter {ch_num}"

                            candidates.append({
                                "chapter_number": ch_num,
                                "chapter_title": title,
                                "physical_page": pno,
                                "font_size": sz,
                                "is_bold": True,
                                "confidence": 0.80,
                                "detection_method": "font_layout",
                                "hierarchy_level": "chapter",
                            })
                            break

        # Deduplicate candidates on same physical page (keep highest confidence)
        unique_by_page: dict[int, dict[str, Any]] = {}
        for c in candidates:
            p = c["physical_page"]
            if p not in unique_by_page or c["confidence"] > unique_by_page[p]["confidence"]:
                unique_by_page[p] = c

        return sorted(unique_by_page.values(), key=lambda x: x["physical_page"])

"""Physical PDF page index vs Printed page number calibration."""

from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass
from typing import Any

import fitz

from app.services.textbook_structure.profiler import DocumentProfile

logger = logging.getLogger(__name__)

# Roman numeral helper
_ROMAN_MAP = {
    "i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000,
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
}


def roman_to_int(s: str) -> int | None:
    s = s.strip()
    if not s or not re.match(r"^[ivxlcdmIVXLCDM]+$", s):
        return None
    val = 0
    prev = 0
    for char in reversed(s):
        curr = _ROMAN_MAP.get(char, 0)
        if curr >= prev:
            val += curr
        else:
            val -= curr
        prev = curr
    return val if val > 0 else None


@dataclass
class OffsetCalibrationResult:
    consensus_offset: int  # physical_page - printed_page
    confidence: float
    method: str  # "margin_consensus", "toc_heading_match", "fallback"
    evidence_count: int
    has_roman_prelims: bool = False
    roman_prelim_count: int = 0


class PageOffsetCalibrator:
    """Robust multi-evidence calibrator for physical PDF page index to printed page offset."""

    @staticmethod
    def calibrate(
        doc: fitz.Document,
        profile: DocumentProfile,
        toc_entries: list[dict[str, Any]] | None = None,
        candidate_headings: list[dict[str, Any]] | None = None,
    ) -> OffsetCalibrationResult:
        total = len(doc)
        if total <= 1:
            return OffsetCalibrationResult(consensus_offset=0, confidence=1.0, method="fallback", evidence_count=0)

        # 1. Check for Roman numerals in preliminary pages (pages 1 to 15)
        roman_count = 0
        for pno in range(min(15, total)):
            text = doc[pno].get_text().strip()
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            for ln in lines[:3] + lines[-3:]:
                if roman_to_int(ln) is not None and len(ln) <= 5:
                    roman_count += 1
                    break
        has_roman = roman_count >= 2

        # 2. Margin-based printed page number scanning across body pages
        # Sample between physical page 5 and min(90, total - 10)
        start_scan = min(10, total // 3)
        end_scan = min(total - 5, max(start_scan + 10, 80))
        sample_pages = [p for p in range(start_scan, end_scan, 2)]

        margin_offsets: list[int] = []
        for pno in sample_pages:
            page = doc[pno]
            h = page.rect.height
            w = page.rect.width

            # Define search rectangles based on profile preference or test both
            rects = []
            if profile.margin_page_number_position == "bottom":
                rects.append(fitz.Rect(0, max(0, h - 75), w, h))
            elif profile.margin_page_number_position == "top":
                rects.append(fitz.Rect(0, 0, w, min(70, h * 0.12)))
            else:
                rects.extend([fitz.Rect(0, max(0, h - 75), w, h), fitz.Rect(0, 0, w, min(70, h * 0.12))])

            for rect in rects:
                margin_txt = page.get_text("text", clip=rect).strip()
                for ln in margin_txt.split("\n"):
                    ln_s = ln.strip()
                    if ln_s.isdigit():
                        p_print = int(ln_s)
                        if 1 <= p_print < total:
                            diff = pno - p_print
                            if -5 <= diff <= 50:  # Sensible offset range
                                margin_offsets.append(diff)
                                break

        if margin_offsets:
            c = collections.Counter(margin_offsets)
            mode_offset, mode_count = c.most_common(1)[0]
            ratio = mode_count / len(margin_offsets)
            if mode_count >= 4 and ratio >= 0.50:
                return OffsetCalibrationResult(
                    consensus_offset=mode_offset,
                    confidence=min(0.98, 0.70 + 0.30 * ratio),
                    method="margin_consensus",
                    evidence_count=mode_count,
                    has_roman_prelims=has_roman,
                    roman_prelim_count=roman_count,
                )

        # 3. Cross-validate TOC entries with body headings (excluding appendix pages)
        toc_heading_offsets: list[int] = []
        if toc_entries and candidate_headings:
            # Main body limit: exclude candidate headings in last 20% of pages if possible
            main_body_limit = int(0.80 * total)
            for toc in toc_entries:
                p_print = toc.get("printed_start_page") or toc.get("printed_page")
                if not isinstance(p_print, int):
                    continue
                c_num = str(toc.get("chapter_number", "")).strip()
                c_title = str(toc.get("chapter_title", "")).strip().lower()

                for bh in candidate_headings:
                    bh_pno = bh.get("physical_page", 0)
                    if bh_pno > main_body_limit and total > 50:
                        continue  # Avoid appendix contamination

                    bh_num = str(bh.get("chapter_number", "")).strip()
                    bh_title = str(bh.get("chapter_title", "")).strip().lower()

                    is_match = False
                    if c_num and bh_num and c_num == bh_num:
                        is_match = True
                    elif len(c_title) > 5 and len(bh_title) > 5 and (c_title in bh_title or bh_title in c_title):
                        is_match = True

                    if is_match:
                        diff = bh_pno - p_print
                        if 0 <= diff <= 40:
                            toc_heading_offsets.append(diff)

        if toc_heading_offsets:
            c = collections.Counter(toc_heading_offsets)
            mode_offset, mode_count = c.most_common(1)[0]
            return OffsetCalibrationResult(
                consensus_offset=mode_offset,
                confidence=0.88,
                method="toc_heading_match",
                evidence_count=mode_count,
                has_roman_prelims=has_roman,
                roman_prelim_count=roman_count,
            )

        # 4. Fallback if margin had any consensus
        if margin_offsets:
            mode_offset, mode_count = collections.Counter(margin_offsets).most_common(1)[0]
            return OffsetCalibrationResult(
                consensus_offset=mode_offset,
                confidence=0.65,
                method="margin_consensus_weak",
                evidence_count=mode_count,
                has_roman_prelims=has_roman,
                roman_prelim_count=roman_count,
            )

        # 5. Default fallback based on Roman prelims count or zero
        default_offset = max(0, roman_count) if has_roman else 0
        return OffsetCalibrationResult(
            consensus_offset=default_offset,
            confidence=0.50,
            method="fallback",
            evidence_count=0,
            has_roman_prelims=has_roman,
            roman_prelim_count=roman_count,
        )

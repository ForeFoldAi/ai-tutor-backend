"""Unified Textbook Structure Detection and Segmentation Engine."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import fitz

from app.services.textbook_structure.boundary_validator import BoundaryValidator
from app.services.textbook_structure.heading_detector import AdaptiveHeadingDetector
from app.services.textbook_structure.hierarchy_resolver import HierarchyResolver, ResolvedChapterNode
from app.services.textbook_structure.offset_calibrator import PageOffsetCalibrator
from app.services.textbook_structure.profiler import DocumentProfile, profile_document
from app.services.textbook_structure.text_cleaner import DocumentTextCleaner, TextCleaningStats
from app.services.textbook_structure.toc_detector import BACK_MATTER_RE, TocEntry, UniversalTocDetector

logger = logging.getLogger(__name__)


_FILENAME_BOOKMARK = re.compile(r"\.pdf$|^\d+\s*[_-]|^(unit|lesson|chapter)[-_ ]?\d+$", re.I)


def _page_heading(page: fitz.Page) -> str | None:
    """Largest-font text on the page (the chapter title on a chapter's first page).

    Consecutive lines in that font size are joined, so titles wrapped over two lines stay whole.
    """
    lines: list[tuple[float, str]] = []
    h = page.rect.height
    for block in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT).get("blocks", []):
        for line in block.get("lines", []):
            # Rotated lines are watermarks; the top/bottom margins hold running headers and footers.
            if abs(line["dir"][1]) > 0.05 or line["bbox"][3] < 0.06 * h or line["bbox"][1] > 0.92 * h:
                continue
            # Spans carry their own spacing; joining with ' ' splits small caps ('W HAT IS').
            text = "".join(s["text"] for s in line["spans"]).strip()
            size = max((s["size"] for s in line["spans"]), default=0)
            if 1 <= len(text) <= 80:
                lines.append((size, text))
    sizes = [sz for sz, t in lines if re.search(r"[A-Za-z]{3}", t)]
    if not sizes:
        return None
    top = max(sizes)
    parts: list[str] = []
    for sz, text in lines:
        if abs(sz - top) <= 0.5:
            parts.append(text)
        elif parts:
            break
    title = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return title[:120] or None


def _clean_filename_title(title: str) -> str:
    """'03_Varsham_Yechhata nundi vastundi 2022 - Final - EM.pdf' -> 'Varsham Yechhata nundi vastundi'."""
    t = re.sub(r"\.pdf$", "", title, flags=re.I)
    t = re.sub(r"^\d+\s*[_\-]+\s*", "", t)
    t = re.sub(r"(\s*[-_]?\s*(20\d\d|final|em|tm|v\d+))+\s*$", "", t, flags=re.I)
    return t.replace("_", " ").strip(" -") or title


def parse_pdf_outline(doc: fitz.Document) -> list[TocEntry]:
    """Extract chapters from digital PDF outline / bookmarks (doc.get_toc()).

    Merged PDFs carry one top-level bookmark per source file ('01_Mana Aaharam-EM.pdf') plus
    nested copies that all point at page 1; only level-1 entries are used, front-matter files are
    skipped, and file-name titles are replaced by the heading printed on the start page.
    """
    entries = []
    last_page = 0
    for item in doc.get_toc() or []:
        if len(item) < 3 or item[0] != 1:
            continue
        raw, p1 = str(item[1]).strip(), int(item[2])
        if p1 <= 0 or p1 > len(doc) or p1 < last_page:
            continue  # out of range or broken link (e.g. points back to page 1)
        last_page = p1
        if re.search(r"preface|contents|cover|copyright|prelim|initial\s*pages", raw, re.I):
            continue

        from_file = bool(_FILENAME_BOOKMARK.search(raw))
        heading = _page_heading(doc[p1 - 1]) if from_file else None
        # A page whose title is a picture yields an exercise sentence instead; the file name is better then.
        if heading and (len(heading.split()) > 10 or re.search(r"\s\d+\.\s", heading)):
            heading = None
        title = (heading or _clean_filename_title(raw)) if from_file else raw
        num = re.search(r"(\d+)", raw) if from_file else None
        m = None if from_file else re.match(
            r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*)$", title, re.I
        )
        if m:
            ch_num, title = m.group(1), m.group(2).strip() or title
        else:
            ch_num = num.group(1).lstrip("0") or "0" if num else str(len(entries) + 1)

        entries.append(
            TocEntry(
                chapter_number=ch_num,
                chapter_title=title,
                hierarchy_level="appendix" if BACK_MATTER_RE.match(title) else "chapter",
                printed_start_page=max(0, p1 - 1),
                printed_end_page=None,
                source_toc_page=0,
                # File-name bookmarks are a guess; keep them below the review threshold.
                confidence=0.75 if from_file else 0.95,
                detection_method="pdf_bookmark",
            )
        )
    return entries


def detect_textbook_structure(pdf_path: str) -> dict[str, Any]:
    """
    Main universal textbook structure detection pipeline:
    1. Adaptive Document Profiling (fonts, text density, watermarks, headers/footers, bookmarks).
    2. Multi-Strategy Table of Contents Extraction (tabular, multi-column, dotted, page ranges).
    3. Digital PDF Bookmark Tree Extraction.
    4. Adaptive Typographic Body Heading Scanning.
    5. Multi-evidence Physical-to-Printed Page Offset Calibration.
    6. Multi-tiered Hierarchy Resolution (Units, Chapters, Readings, Appendices).
    7. Gap-Free, Non-Overlapping Boundary Resolution & Anomaly Validation.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Textbook PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        profile = profile_document(doc)

        # Signal 1: Digital Bookmarks
        bookmark_entries = parse_pdf_outline(doc)

        # Signal 2: Printed Table of Contents
        toc_entries = UniversalTocDetector.parse_toc(doc)

        # Offset from printed margin page numbers first; scanning every page for headings is the
        # slowest step (~25 ms/page), so it only runs when margins or the TOC can't answer.
        combined_toc_for_offset = [t.__dict__ for t in (toc_entries or bookmark_entries)]
        offset_res = PageOffsetCalibrator.calibrate(doc, profile, combined_toc_for_offset)
        if offset_res.method != "margin_consensus" or not combined_toc_for_offset:
            body_headings = AdaptiveHeadingDetector.scan_headings(doc, profile)
            offset_res = PageOffsetCalibrator.calibrate(doc, profile, combined_toc_for_offset, body_headings)
        else:
            # Only units with readings need headings (to place each reading inside its unit).
            unit_pages = sorted({
                p
                for e in toc_entries if e.sub_readings
                for p in range(
                    max(0, e.printed_start_page + offset_res.consensus_offset),
                    min(total_pages, (e.printed_end_page or e.printed_start_page) + offset_res.consensus_offset + 1),
                )
            })
            body_headings = AdaptiveHeadingDetector.scan_headings(doc, profile, unit_pages)

        # Combine TOC candidates (prefer printed TOC with ranges, augment with bookmarks)
        effective_toc: list[TocEntry] = []
        if toc_entries:
            effective_toc.extend(toc_entries)
        elif bookmark_entries:
            # If no printed TOC, bookmarks are used with physical start pages
            for b in bookmark_entries:
                # bookmark p1 was 1-indexed physical
                b.printed_start_page = max(1, b.printed_start_page - offset_res.consensus_offset)
                effective_toc.append(b)

        # Hierarchy Resolution
        raw_nodes = HierarchyResolver.resolve_hierarchy(
            effective_toc,
            body_headings,
            offset_res.consensus_offset,
            total_pages,
        )

        # Boundary Validation & Resolution
        main_chapters, non_chapters, warnings, needs_review = BoundaryValidator.validate_and_resolve(
            raw_nodes,
            total_pages,
        )

        # Calculate overall document confidence
        if main_chapters:
            avg_conf = sum(ch.confidence_score for ch in main_chapters) / len(main_chapters)
            # Downgrade if review warnings exist
            if warnings:
                avg_conf = max(0.50, avg_conf - 0.10)
        else:
            avg_conf = 0.30
            needs_review = True

        return {
            "total_pages": total_pages,
            "pdf_type": profile.pdf_type,
            "page_offset": offset_res.consensus_offset,
            "offset_confidence": offset_res.confidence,
            "detection_method": "universal_hybrid",
            "confidence_score": round(avg_conf, 2),
            "needs_manual_review": needs_review,
            "review_warnings": warnings,
            "toc_found": bool(toc_entries or bookmark_entries),
            "toc_page_numbers": UniversalTocDetector.find_toc_pages(doc),
            "chapters": [c.to_dict() for c in main_chapters],
            "non_chapter_sections": [nc.to_dict() for nc in non_chapters],
            "profile": {
                "total_pages": profile.total_pages,
                "pdf_type": profile.pdf_type,
                "baseline_font_size": profile.baseline_font_size,
                "has_native_bookmarks": profile.has_native_bookmarks,
                "bookmark_count": profile.bookmark_count,
                "repeated_footers": profile.repeated_footers,
                "detected_watermarks": profile.detected_watermarks,
                "margin_page_number_position": profile.margin_page_number_position,
            },
        }
    finally:
        doc.close()

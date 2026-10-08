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
from app.services.textbook_structure.toc_detector import TocEntry, UniversalTocDetector

logger = logging.getLogger(__name__)


def parse_pdf_outline(doc: fitz.Document) -> list[TocEntry]:
    """Extract chapters from digital PDF outline / bookmarks (doc.get_toc())."""
    entries = []
    toc = doc.get_toc() or []
    for item in toc:
        if len(item) < 3:
            continue
        lvl, title, p1 = item[0], str(item[1]).strip(), int(item[2])
        if p1 <= 0 or p1 > len(doc):
            continue
        if lvl <= 2:
            # Check for non-chapter bookmarks
            if any(kw in title.lower() for kw in ["preface", "contents", "cover", "copyright", "prelim"]):
                continue

            m = re.match(r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*)$", title, re.I)
            if m:
                ch_num = m.group(1)
                ch_title = m.group(2).strip()
            else:
                ch_num = str(len(entries) + 1)
                ch_title = title

            entries.append(
                TocEntry(
                    chapter_number=ch_num,
                    chapter_title=ch_title or title,
                    hierarchy_level="chapter",
                    printed_start_page=max(0, p1 - 1),
                    printed_end_page=None,
                    source_toc_page=0,
                    confidence=0.95,
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

        # Signal 3: Body Heading Scanner
        body_headings = AdaptiveHeadingDetector.scan_headings(doc, profile)

        # Multi-evidence Offset Calibration
        combined_toc_for_offset = [t.__dict__ for t in (toc_entries or bookmark_entries)]
        offset_res = PageOffsetCalibrator.calibrate(doc, profile, combined_toc_for_offset, body_headings)

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

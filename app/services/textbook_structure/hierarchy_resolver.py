"""Textbook hierarchy resolution (Parts, Units, Chapters, Readings, Sections)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.textbook_structure.toc_detector import TocEntry

logger = logging.getLogger(__name__)


@dataclass
class ResolvedChapterNode:
    chapter_number: str
    chapter_title: str
    hierarchy_level: str  # "unit", "chapter", "reading", "appendix", "front_matter", "back_matter"
    start_pdf_page: int
    end_pdf_page: int
    printed_start_page: str | None = None
    printed_end_page: str | None = None
    is_non_chapter_section: bool = False
    section_type: str = "chapter"
    detection_method: str = "toc_body_match"
    confidence_score: float = 0.90
    confidence_flags: list[str] = field(default_factory=list)
    parent_number: str | None = None
    sub_chapters: list["ResolvedChapterNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "hierarchy_level": self.hierarchy_level,
            "start_pdf_page": self.start_pdf_page,
            "end_pdf_page": self.end_pdf_page,
            "printed_start_page": self.printed_start_page,
            "printed_end_page": self.printed_end_page,
            "is_non_chapter_section": self.is_non_chapter_section,
            "section_type": self.section_type,
            "detection_method": self.detection_method,
            "confidence_score": self.confidence_score,
            "confidence_flags": self.confidence_flags,
            "parent_number": self.parent_number,
            "sub_chapters": [sc.to_dict() for sc in self.sub_chapters],
        }


class HierarchyResolver:
    """Builds and resolves parent-child chapter hierarchies."""

    @classmethod
    def resolve_hierarchy(
        cls,
        toc_entries: list[TocEntry],
        body_headings: list[dict[str, Any]],
        page_offset: int,
        total_pages: int,
    ) -> list[ResolvedChapterNode]:
        """Convert TOC entries and body headings into structured hierarchical chapter nodes."""
        nodes: list[ResolvedChapterNode] = []

        if not toc_entries:
            # Fall back to body headings as flat chapters
            for bh in body_headings:
                p_phys = bh["physical_page"]
                nodes.append(
                    ResolvedChapterNode(
                        chapter_number=str(bh.get("chapter_number", len(nodes) + 1)),
                        chapter_title=bh.get("chapter_title", f"Chapter {len(nodes) + 1}"),
                        hierarchy_level=bh.get("hierarchy_level", "chapter"),
                        start_pdf_page=p_phys,
                        end_pdf_page=p_phys,
                        printed_start_page=str(max(1, p_phys - page_offset + 1)),
                        detection_method=bh.get("detection_method", "layout_heading"),
                        confidence_score=bh.get("confidence", 0.75),
                    )
                )
            return nodes

        # Process TOC entries with calibrated page offset
        for toc in toc_entries:
            p_print_start = toc.printed_start_page
            p_phys_start = max(0, min(total_pages - 1, p_print_start + page_offset))

            p_print_end = toc.printed_end_page
            p_phys_end = (
                max(p_phys_start, min(total_pages - 1, p_print_end + page_offset))
                if p_print_end is not None
                else p_phys_start
            )

            is_non_ch = toc.hierarchy_level in ("appendix", "front_matter", "back_matter")
            sec_type = "appendix" if toc.hierarchy_level == "appendix" else "chapter"

            parent_node = ResolvedChapterNode(
                chapter_number=str(toc.chapter_number),
                chapter_title=toc.chapter_title,
                hierarchy_level=toc.hierarchy_level,
                start_pdf_page=p_phys_start,
                end_pdf_page=p_phys_end,
                printed_start_page=str(p_print_start),
                printed_end_page=str(p_print_end) if p_print_end else None,
                is_non_chapter_section=is_non_ch,
                section_type=sec_type,
                detection_method=toc.detection_method,
                confidence_score=toc.confidence,
            )

            # Check for sub-readings (e.g. Reading A, Reading B, Reading C)
            if toc.sub_readings:
                # Find matching body headings within this Unit's page range
                unit_headings = [
                    bh for bh in body_headings
                    if p_phys_start <= bh["physical_page"] <= p_phys_end
                ]

                for idx, sr_text in enumerate(toc.sub_readings):
                    sr_letter = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
                    # Check if body heading matches this reading
                    matching_bh = next(
                        (bh for bh in unit_headings if bh.get("chapter_number") == sr_letter or sr_letter in bh.get("chapter_title", "")),
                        None
                    )
                    sub_start = matching_bh["physical_page"] if matching_bh else p_phys_start
                    sub_title = sr_text if sr_text.startswith("Reading") else f"Reading {sr_text}"

                    sub_node = ResolvedChapterNode(
                        chapter_number=f"{toc.chapter_number}.{sr_letter}",
                        chapter_title=sub_title,
                        hierarchy_level="reading",
                        start_pdf_page=sub_start,
                        end_pdf_page=p_phys_end,  # Adjusted during boundary validation
                        printed_start_page=str(max(1, sub_start - page_offset + 1)),
                        parent_number=str(toc.chapter_number),
                        detection_method="toc_sub_reading",
                        confidence_score=0.90,
                    )
                    parent_node.sub_chapters.append(sub_node)

            nodes.append(parent_node)

        return nodes

    @classmethod
    def flatten_for_ingestion(
        cls,
        nodes: list[ResolvedChapterNode],
        granularity: str = "chapter",  # "unit", "chapter", "reading"
    ) -> list[ResolvedChapterNode]:
        """Flatten or select nodes depending on chosen ingestion granularity."""
        if granularity in ("unit", "chapter") or not any(n.sub_chapters for n in nodes):
            return nodes

        # If granularity is "reading", replace units that have sub_readings with their children
        flattened: list[ResolvedChapterNode] = []
        for n in nodes:
            if n.sub_chapters:
                # Adjust consecutive end pages for sub-readings
                for idx, child in enumerate(n.sub_chapters):
                    if idx + 1 < len(n.sub_chapters):
                        child.end_pdf_page = max(child.start_pdf_page, n.sub_chapters[idx + 1].start_pdf_page - 1)
                    else:
                        child.end_pdf_page = n.end_pdf_page
                    flattened.append(child)
            else:
                flattened.append(n)

        return flattened

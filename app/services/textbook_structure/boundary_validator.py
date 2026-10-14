"""Boundary resolution, gap handling, overlap elimination, and review validation."""

from __future__ import annotations

import logging
from typing import Any

from app.services.textbook_structure.hierarchy_resolver import ResolvedChapterNode

logger = logging.getLogger(__name__)


class BoundaryValidator:
    """Validates chapter boundaries, calculates non-overlapping spans, and identifies anomalies."""

    @classmethod
    def validate_and_resolve(
        cls,
        nodes: list[ResolvedChapterNode],
        total_pages: int,
    ) -> tuple[list[ResolvedChapterNode], list[ResolvedChapterNode], list[str], bool]:
        """
        Returns:
            chapters: Validated main instructional chapters.
            non_chapter_sections: Front matter, appendices, back matter.
            warnings: List of warning messages.
            needs_manual_review: Flag indicating if admin review is recommended.
        """
        warnings: list[str] = []
        needs_review = False

        if not nodes:
            # Fallback single chapter
            fallback = ResolvedChapterNode(
                chapter_number="1",
                chapter_title="Complete Document",
                hierarchy_level="chapter",
                start_pdf_page=0,
                end_pdf_page=max(0, total_pages - 1),
                confidence_score=0.40,
                confidence_flags=["No chapter markers found; defaulting to full document."],
            )
            return [fallback], [], ["No chapters detected; whole document mapped as single chapter."], True

        # Sort nodes by start_pdf_page
        sorted_nodes = sorted(nodes, key=lambda n: (n.start_pdf_page, n.end_pdf_page))

        # Separate main chapters from non-chapter sections (appendices, front matter)
        main_chapters: list[ResolvedChapterNode] = []
        non_chapter_sections: list[ResolvedChapterNode] = []

        for n in sorted_nodes:
            if n.is_non_chapter_section or n.hierarchy_level in ("appendix", "front_matter", "back_matter"):
                n.is_non_chapter_section = True
                non_chapter_sections.append(n)
            else:
                main_chapters.append(n)

        # 1. Front Matter Check
        first_ch_start = main_chapters[0].start_pdf_page if main_chapters else 0
        if first_ch_start > 0:
            front_matter = ResolvedChapterNode(
                chapter_number="FM",
                chapter_title="Preliminary Pages (Cover, Preface, Contents)",
                hierarchy_level="front_matter",
                start_pdf_page=0,
                end_pdf_page=first_ch_start - 1,
                is_non_chapter_section=True,
                section_type="front_matter",
                detection_method="boundary_inference",
                confidence_score=1.0,
            )
            non_chapter_sections.insert(0, front_matter)

        # 2. Main Chapter Boundary Calculation
        for idx, ch in enumerate(main_chapters):
            # If end_pdf_page was not explicitly provided from TOC range, compute from next chapter start
            if ch.end_pdf_page <= ch.start_pdf_page or ch.end_pdf_page >= total_pages:
                if idx + 1 < len(main_chapters):
                    next_start = main_chapters[idx + 1].start_pdf_page
                    ch.end_pdf_page = max(ch.start_pdf_page, next_start - 1)
                elif non_chapter_sections and any(nc.start_pdf_page > ch.start_pdf_page for nc in non_chapter_sections):
                    # Next is an appendix
                    next_app_start = min(nc.start_pdf_page for nc in non_chapter_sections if nc.start_pdf_page > ch.start_pdf_page)
                    ch.end_pdf_page = max(ch.start_pdf_page, next_app_start - 1)
                else:
                    ch.end_pdf_page = total_pages - 1

            # Bounds clamping
            ch.start_pdf_page = max(0, min(total_pages - 1, ch.start_pdf_page))
            ch.end_pdf_page = max(ch.start_pdf_page, min(total_pages - 1, ch.end_pdf_page))

        # 3. Resolve Overlaps among main chapters
        for idx in range(len(main_chapters) - 1):
            curr_ch = main_chapters[idx]
            next_ch = main_chapters[idx + 1]

            if curr_ch.end_pdf_page >= next_ch.start_pdf_page:
                # Overlap detected
                curr_ch.end_pdf_page = max(curr_ch.start_pdf_page, next_ch.start_pdf_page - 1)
                warnings.append(
                    f"Resolved overlap between '{curr_ch.chapter_title}' and '{next_ch.chapter_title}'."
                )

        # 4. Appendix Boundary Calculation
        for idx, nc in enumerate(non_chapter_sections):
            if nc.hierarchy_level == "front_matter":
                continue
            if nc.end_pdf_page <= nc.start_pdf_page:
                if idx + 1 < len(non_chapter_sections):
                    nc.end_pdf_page = max(nc.start_pdf_page, non_chapter_sections[idx + 1].start_pdf_page - 1)
                else:
                    nc.end_pdf_page = total_pages - 1
            nc.start_pdf_page = max(0, min(total_pages - 1, nc.start_pdf_page))
            nc.end_pdf_page = max(nc.start_pdf_page, min(total_pages - 1, nc.end_pdf_page))

        # 5. Quality and Anomaly Checks
        for ch in main_chapters:
            page_span = ch.end_pdf_page - ch.start_pdf_page + 1
            if page_span < 2:
                ch.confidence_flags.append("Very short chapter (less than 2 pages).")
                warnings.append(f"Chapter '{ch.chapter_title}' spans only {page_span} page(s).")
                needs_review = True
            elif page_span > 80:
                ch.confidence_flags.append("Unusually long chapter (greater than 80 pages).")
                warnings.append(f"Chapter '{ch.chapter_title}' spans {page_span} pages.")
                needs_review = True

            if ch.confidence_score < 0.75:
                needs_review = True
                warnings.append(f"Low confidence ({ch.confidence_score:.2f}) for chapter '{ch.chapter_title}'.")

        return main_chapters, non_chapter_sections, warnings, needs_review

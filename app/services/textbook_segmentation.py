"""Textbook structure detection, chapter segmentation, and isolated indexing.

Implements hybrid chapter detection combining:
1. PDF TOC / outline bookmarks (doc.get_toc()).
2. Printed TOC preliminary page detection and regex parsing.
3. Body typographic layout signals (PyMuPDF span font size, bold flags, y-position).
4. Printed page vs physical PDF page offset reconciliation.
5. Negative filtering for captions, exercises, and body text references.
6. Optional LLM fallback for ambiguous candidates.
7. Chapter PDF slicing and isolated indexing linking into existing TextbookUpload.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Roman numeral conversion helper
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


# Word to number mapping for chapter names
_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def parse_chapter_number_str(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.isdigit():
        return str(int(cleaned))
    as_roman = roman_to_int(cleaned)
    if as_roman is not None:
        return str(as_roman)
    as_word = _WORD_NUMS.get(cleaned.lower())
    if as_word is not None:
        return str(as_word)
    return cleaned


def detect_pdf_type(doc: fitz.Document) -> str:
    """Classify PDF as text_based, scanned, or mixed."""
    total_pages = len(doc)
    if total_pages == 0:
        return "text_based"

    low_text_count = 0
    # Sample up to first 25 pages + spread across doc
    sample_indices = list(range(min(15, total_pages)))
    if total_pages > 15:
        step = max(1, total_pages // 10)
        sample_indices.extend(range(15, total_pages, step))
    sample_indices = sorted(list(set(sample_indices)))

    for pno in sample_indices:
        text = doc[pno].get_text() or ""
        if len(text.strip()) < 50:
            low_text_count += 1

    ratio = low_text_count / max(1, len(sample_indices))
    if ratio >= 0.85:
        return "scanned"
    elif ratio >= 0.20:
        return "mixed"
    return "text_based"


def parse_pdf_outline(doc: fitz.Document) -> list[dict[str, Any]]:
    """Extract chapters from PDF outline / bookmarks (doc.get_toc())."""
    entries = []
    toc = doc.get_toc() or []
    # toc items: [lvl, title, page_1_indexed]
    for item in toc:
        if len(item) < 3:
            continue
        lvl, title, p1 = item[0], str(item[1]).strip(), int(item[2])
        if p1 <= 0 or p1 > len(doc):
            continue
        # Filter top-level or chapter-level items (lvl <= 2)
        if lvl <= 2:
            m = re.match(
                r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*)$",
                title,
                re.IGNORECASE,
            )
            ch_num = m.group(1) if m else str(len(entries) + 1)
            ch_title = m.group(2).strip() if m else title
            entries.append({
                "chapter_number": parse_chapter_number_str(ch_num),
                "chapter_title": ch_title or title,
                "target_page": p1 - 1,  # 0-indexed
                "printed_page": None,
                "source": "outline",
            })
    return entries


def parse_printed_toc(doc: fitz.Document) -> list[dict[str, Any]]:
    """Scan preliminary pages (0 to 30) for a Table of Contents and parse entries."""
    entries = []
    max_scan = min(30, len(doc))
    toc_pages = []

    for pno in range(max_scan):
        text = doc[pno].get_text() or ""
        first_few_lines = "\n".join(text.split("\n")[:8]).lower()
        if any(kw in first_few_lines for kw in ["contents", "table of contents", "index", "विषय-सूची", "विषय सूची"]):
            toc_pages.append(pno)

    if not toc_pages:
        return entries

    # Parse lines inside identified TOC pages
    line_pattern = re.compile(
        r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*?)\s+[\.\s_]*(\d+|[ivxlcdm]+)$",
        re.IGNORECASE,
    )
    dotted_pattern = re.compile(
        r"^(.*?)\s+[\.·\s_]{3,}\s*(\d+|[ivxlcdm]+)$",
        re.IGNORECASE,
    )

    for pno in toc_pages:
        lines = [ln.strip() for ln in doc[pno].get_text().split("\n") if ln.strip()]
        for line in lines:
            # Check standard chapter line
            m = line_pattern.match(line)
            if m:
                ch_num_raw, title, page_str = m.group(1), m.group(2).strip(), m.group(3).strip()
                p_int = roman_to_int(page_str) if not page_str.isdigit() else int(page_str)
                entries.append({
                    "chapter_number": parse_chapter_number_str(ch_num_raw),
                    "chapter_title": title,
                    "printed_page": p_int,
                    "source": "printed_toc",
                    "toc_pno": pno,
                })
                continue

            # Check dotted leader line
            m2 = dotted_pattern.match(line)
            if m2:
                title, page_str = m2.group(1).strip(), m2.group(2).strip()
                p_int = roman_to_int(page_str) if not page_str.isdigit() else int(page_str)
                # Extract chapter number if present in title
                ch_match = re.match(r"^(?:(?:Chapter|Unit|Lesson|अध्याय)\s*)?(\d+|[IVXLCDM]+|[A-Z])[\.:\s\-]+(.*)$", title, re.I)
                if ch_match:
                    ch_num_raw = ch_match.group(1)
                    title = ch_match.group(2).strip()
                else:
                    ch_num_raw = str(len(entries) + 1)
                entries.append({
                    "chapter_number": parse_chapter_number_str(ch_num_raw),
                    "chapter_title": title,
                    "printed_page": p_int,
                    "source": "printed_toc",
                    "toc_pno": pno,
                })

    return entries


def compute_baseline_font_size(doc: fitz.Document) -> float:
    """Compute the modal body font size across the document."""
    sizes: list[float] = []
    step = max(1, len(doc) // 20)
    for pno in range(0, len(doc), step):
        d = doc[pno].get_text("dict")
        for b in d.get("blocks", []):
            if "lines" in b:
                for line in b["lines"]:
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if len(txt) > 5:
                            sizes.append(round(span.get("size", 10.0), 1))
    if not sizes:
        return 10.0
    counter = collections.Counter(sizes)
    return counter.most_common(1)[0][0]


def scan_body_headings(doc: fitz.Document, base_font_size: float) -> list[dict[str, Any]]:
    """Scan document body pages for prominent chapter headings using layout & regex signals."""
    candidates = []
    heading_regex = re.compile(
        r"^(?:CHAPTER|Chapter|UNIT|Unit|LESSON|Lesson|अध्याय)\s*(\d+|[IVXLCDM]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b",
        re.IGNORECASE,
    )
    # Negative filters for figure captions, exercises, and body text references
    negative_regex = re.compile(
        r"^(?:fig(?:ure)?\s*\d+|table\s*\d+|diagram\s*\d+|exercise|exercises|questions|review|in\s+chapter|refer\s+to|as\s+seen\s+in|source:)",
        re.IGNORECASE,
    )

    for pno in range(len(doc)):
        page = doc[pno]
        h = page.rect.height
        d = page.get_text("dict")
        page_text = page.get_text() or ""

        # Check block by block
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

                    # Must be in upper 65% of the page and not in extreme header band
                    if y0 < 15 or y0 > 0.65 * h:
                        continue

                    # Reject false positives
                    if negative_regex.search(txt):
                        continue

                    # Case A: Explicit Chapter regex match
                    m = heading_regex.match(txt)
                    if m:
                        ch_num_raw = m.group(1)
                        # Determine title: check remainder of line, or next lines/spans
                        remainder = txt[m.end():].lstrip(" :.-")
                        title = remainder
                        if not title:
                            # Look at next lines in same block or next block
                            title_candidates = []
                            for subsequent_span in line.get("spans", []):
                                sub_txt = subsequent_span.get("text", "").strip()
                                if sub_txt and sub_txt != txt:
                                    title_candidates.append(sub_txt)
                            if not title_candidates and b_idx + 1 < len(blocks):
                                next_b = blocks[b_idx + 1]
                                for nl in next_b.get("lines", []):
                                    for ns in nl.get("spans", []):
                                        st = ns.get("text", "").strip()
                                        if st and len(st) > 2:
                                            title_candidates.append(st)
                            title = " ".join(title_candidates[:2])

                        candidates.append({
                            "chapter_number": parse_chapter_number_str(ch_num_raw),
                            "chapter_title": title or f"Chapter {ch_num_raw}",
                            "physical_page": pno,
                            "font_size": sz,
                            "is_bold": bool(flags & 2),
                            "confidence": 0.85 if sz >= 1.2 * base_font_size else 0.70,
                            "detection_method": "layout_heading",
                        })
                        break

                    # Case B: Typographically prominent standalone chapter header (e.g. 72pt number or > 1.8x base font)
                    if sz >= 1.8 * base_font_size and len(txt) <= 60 and not negative_regex.search(txt):
                        is_likely_chapter = False
                        ch_num = None
                        if txt.isdigit() and int(txt) < 50:
                            is_likely_chapter = True
                            ch_num = txt
                        elif heading_regex.search(page_text[:300]):
                            is_likely_chapter = True
                            m_page = heading_regex.search(page_text[:300])
                            ch_num = m_page.group(1) if m_page else str(len(candidates) + 1)

                        if is_likely_chapter:
                            # Extract true title from highest font size non-digit title spans on this page
                            title_spans = []
                            for ob in blocks:
                                for ol in ob.get("lines", []):
                                    for os_span in ol.get("spans", []):
                                        otxt = os_span.get("text", "").strip()
                                        osz = os_span.get("size", 0.0)
                                        if (
                                            osz >= 1.4 * base_font_size
                                            and not otxt.isdigit()
                                            and otxt.upper() not in ("CHAPTER", "UNIT", "LESSON", "अध्याय")
                                            and osz < 65
                                            and not negative_regex.search(otxt)
                                        ):
                                            title_spans.append((osz, otxt))
                            if title_spans:
                                max_sz = max(s[0] for s in title_spans)
                                title = " ".join(s[1] for s in title_spans if abs(s[0] - max_sz) <= 2.5)
                            else:
                                title = txt if not txt.isdigit() else f"Chapter {ch_num}"

                            candidates.append({
                                "chapter_number": parse_chapter_number_str(ch_num or str(len(candidates) + 1)),
                                "chapter_title": title,
                                "physical_page": pno,
                                "font_size": sz,
                                "is_bold": True,
                                "confidence": 0.85,
                                "detection_method": "font_layout",
                            })
                            break
    return candidates


def reconcile_page_offset(
    toc_entries: list[dict[str, Any]],
    body_headings: list[dict[str, Any]],
    total_pages: int,
) -> int:
    """Calculate consensus offset: delta = physical_page - printed_page."""
    offsets = []
    for toc in toc_entries:
        p_print = toc.get("printed_page")
        if p_print is None or not isinstance(p_print, int):
            continue
        c_num = toc.get("chapter_number")
        for bh in body_headings:
            if bh.get("chapter_number") == c_num:
                diff = bh["physical_page"] - p_print
                if 0 <= diff < 50:  # Realistic front matter range
                    offsets.append(diff)

    if offsets:
        counter = collections.Counter(offsets)
        return counter.most_common(1)[0][0]
    # Default reasonable fallback: check if body headings start around page 4-10
    if body_headings and body_headings[0].get("chapter_number") == "1":
        return max(0, body_headings[0]["physical_page"] - 1)
    return 0


def disambiguate_with_llm(page_text: str, candidate_title: str) -> dict[str, Any] | None:
    """Optional LLM-assisted resolution for ambiguous chapter starts."""
    try:
        from app.services.llm_client import get_default_llm_client

        client = get_default_llm_client()
        prompt = (
            "You are a strict textbook parser. Analyze the following page text to determine if it is the true beginning "
            "of a new textbook chapter. Return ONLY valid JSON with keys: "
            '{"is_chapter_start": true/false, "chapter_number": "...", "chapter_title": "..."}\n\n'
            f"Candidate Title: {candidate_title}\n"
            f"Page Text Excerpt:\n{page_text[:1200]}\n"
        )
        resp = client.chat(messages=[{"role": "user", "content": prompt}], temperature=0.0)
        content = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
        # Parse JSON
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as exc:
        logger.warning("LLM disambiguation skipped: %s", exc)
    return None


def detect_textbook_structure(pdf_path: str) -> dict[str, Any]:
    """
    Main detection pipeline:
    Analyzes PDF, detects chapters, page boundaries, Roman numeral preliminaries,
    non-chapter front/back matter, and confidence scores via the unified engine.
    """
    from app.services.textbook_structure import detect_textbook_structure as _detect_engine

    return _detect_engine(pdf_path)



def slice_chapter_pdf(src_pdf_path: str, start_page: int, end_page: int, dst_pdf_path: str) -> None:
    """Extract a slice of pages from src_pdf_path into dst_pdf_path with minimal memory overhead."""
    os.makedirs(os.path.dirname(os.path.abspath(dst_pdf_path)), exist_ok=True)
    src_doc = fitz.open(src_pdf_path)
    total = len(src_doc)
    start = max(0, min(start_page, total - 1))
    end = max(start, min(end_page, total - 1))

    dst_doc = fitz.open()
    dst_doc.insert_pdf(src_doc, from_page=start, to_page=end)
    dst_doc.save(dst_pdf_path, garbage=4, deflate=True)
    dst_doc.close()
    src_doc.close()


def process_textbook_chapter_background(chapter_id: int) -> None:
    """
    Background worker:
    1. Slices chapter page range into an isolated chapter PDF.
    2. Creates/updates a TextbookUpload row linked to this chapter.
    3. Runs pedagogical chunking and vector indexing.
    4. Runs isolated image/figure extraction.
    5. Updates chapter status to COMPLETED.
    """
    from app.core.database import SessionLocal
    from app.modules.catalog.models import (
        ChapterStatusEnum,
        ProcessingStatusEnum,
        Textbook,
        TextbookChapter,
        TextbookUpload,
    )
    from app.services.document_service import process_document
    from app.services.image_service.storage_backend import materialize_textbook_file
    from app.services.vector_service import add_documents_to_store, delete_collection_docs

    db = SessionLocal()
    try:
        ch = db.get(TextbookChapter, chapter_id)
        if not ch:
            logger.error("TextbookChapter %s not found", chapter_id)
            return

        tb = db.get(Textbook, ch.textbook_id)
        if not tb:
            logger.error("Textbook %s not found for chapter %s", ch.textbook_id, chapter_id)
            return

        ch.status = ChapterStatusEnum.PROCESSING
        ch.error_message = None
        db.commit()

        # Materialize source textbook PDF
        local_src_path = materialize_textbook_file(tb.file_path)
        if not os.path.isfile(local_src_path):
            raise FileNotFoundError(f"Source textbook missing: {tb.file_path}")

        # Destination path for isolated chapter PDF
        ch_slug = f"ch_{ch.chapter_number}_{ch.id}"
        slice_key = f"{tb.board.value}/{tb.class_level.value}/{tb.subject_name}/chapters/{ch_slug}.pdf"
        from app.services.image_service.storage_backend import get_document_storage_backend

        # Local destination path
        local_slice_path = os.path.join("uploads", slice_key)
        slice_chapter_pdf(local_src_path, ch.start_pdf_page, ch.end_pdf_page, local_slice_path)

        # Upload to storage backend if S3 or MinIO configured
        with open(local_slice_path, "rb") as f:
            slice_bytes = f.read()
        get_document_storage_backend().save(slice_key, slice_bytes)

        # Create or update linked TextbookUpload row for zero-breaking downstream compatibility
        tu = None
        if ch.textbook_upload_id:
            tu = db.get(TextbookUpload, ch.textbook_upload_id)

        if not tu:
            tu = TextbookUpload(
                file_name=f"{tb.title} - Chapter {ch.chapter_number}",
                board=tb.board,
                class_level=tb.class_level,
                subject_name=tb.subject_name,
                chapter=ch.chapter_title,
                content_type="CHAPTER",
                content_label=f"Chapter {ch.chapter_number}: {ch.chapter_title}",
                file_path=slice_key,
                uploaded_by=tb.uploaded_by,
            )
            db.add(tu)
            db.commit()
            db.refresh(tu)
            ch.textbook_upload_id = tu.id
            db.commit()

        # Ensure idempotency: purge old vectors & images if retried
        collection_name = f"{tb.board}_{tb.class_level}_{tb.subject_name}".replace(" ", "_")
        try:
            delete_collection_docs(collection_name, where_filter={"textbook_upload_id": str(tu.id)})
        except Exception:
            pass

        from app.services.image_service.textbook_image_extraction import purge_textbook_images_disk_and_rows
        purge_textbook_images_disk_and_rows(db, tu.id)
        db.commit()

        # Pedagogical chunking & indexing
        extra_meta = {
            "board": str(tb.board),
            "class_level": str(tb.class_level),
            "subject_name": tb.subject_name,
            "textbook_id": str(tb.id),
            "chapter_id": str(ch.id),
            "parent_id": str(ch.parent_id or ""),
            "hierarchy_level": ch.hierarchy_level or "chapter",
            "textbook_upload_id": str(tu.id),
            "chapter": ch.chapter_title,
            "chapter_number": ch.chapter_number,
            "content_type": ch.hierarchy_level.upper() if ch.hierarchy_level else "CHAPTER",
            "content_label": tu.content_label or "",
            "start_pdf_page": ch.start_pdf_page,
            "end_pdf_page": ch.end_pdf_page,
            "printed_start_page": ch.printed_start_page or "",
            "printed_end_page": ch.printed_end_page or "",
        }
        chunks = process_document(local_slice_path, extra_metadata=extra_meta)
        tu.chunk_count = len(chunks)
        tu.chunk_status = ProcessingStatusEnum.EMBEDDED
        db.commit()

        # Add vectors
        added = add_documents_to_store(chunks, collection_name=collection_name)
        tu.embedding_status = ProcessingStatusEnum.EMBEDDED if added > 0 else ProcessingStatusEnum.FAILED
        tu.ocr_status = ProcessingStatusEnum.EMBEDDED
        db.commit()

        # Isolated Image & Figure extraction
        try:
            from app.services.image_service.textbook_image_extraction import replace_all_images_after_reprocess
            replace_all_images_after_reprocess(db, tu)
        except Exception as img_exc:
            logger.warning("Chapter image extraction failed (non-fatal): %s", img_exc)

        ch.status = ChapterStatusEnum.COMPLETED
        db.commit()
        logger.info(
            "Successfully processed chapter %s (%s): %d chunks, %d vectors",
            ch.id, ch.chapter_title, len(chunks), added
        )
    except Exception as exc:
        logger.exception("Failed to process chapter %s: %s", chapter_id, exc)
        try:
            ch = db.get(TextbookChapter, chapter_id)
            if ch:
                ch.status = ChapterStatusEnum.FAILED
                ch.error_message = str(exc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()

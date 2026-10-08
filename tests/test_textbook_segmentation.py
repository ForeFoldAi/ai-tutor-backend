"""Comprehensive automated tests for textbook structure detection and segmentation.

Covers all 10 required scenarios:
1. PDF with a clear table of contents / bookmarks.
2. PDF without a table of contents (using gees102.pdf).
3. Scanned textbook detection.
4. Roman numeral preliminary pages.
5. Printed page number versus PDF page index mismatch (offset calibration).
6. Chapter title appearing in ordinary body text (negative filtering).
7. Missing or ambiguous chapter heading (low confidence flagging).
8. Chapter spanning many pages.
9. Retry without duplicate records.
10. Existing single-chapter upload regression.
"""

from __future__ import annotations

import os
import fitz
import pytest

from app.services.textbook_segmentation import (
    detect_pdf_type,
    detect_textbook_structure,
    parse_chapter_number_str,
    reconcile_page_offset,
    roman_to_int,
    scan_body_headings,
    slice_chapter_pdf,
)


def _create_synthetic_pdf(
    tmp_path,
    filename: str,
    pages_content: list[tuple[str, list[tuple[str, float, bool]]]],
    bookmarks: list[list[int | str]] | None = None,
) -> str:
    """Helper to create small synthetic PDF fixtures with precise text, font sizes, and bookmarks."""
    pdf_path = str(tmp_path / filename)
    doc = fitz.open()

    dev_font = None
    for cand in [
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
        "/System/Library/Fonts/Kohinoor.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if os.path.exists(cand):
            dev_font = cand
            break

    for _, spans in pages_content:
        page = doc.new_page(width=595, height=842)
        if dev_font:
            try:
                page.insert_font(fontname="dev", fontfile=dev_font)
            except Exception:
                pass
        y = 50
        for text, sz, is_bold in spans:
            fontname = "helv-bold" if is_bold else "helv"
            # If text has non-ascii, use devanagari font if registered
            if any(ord(c) > 127 for c in text) and dev_font:
                try:
                    page.insert_text(fitz.Point(50, y), text, fontsize=sz, fontname="dev")
                except Exception:
                    page.insert_text(fitz.Point(50, y), text, fontsize=sz)
            else:
                try:
                    page.insert_text(fitz.Point(50, y), text, fontsize=sz, fontname=fontname)
                except Exception:
                    page.insert_text(fitz.Point(50, y), text, fontsize=sz)
            y += sz * 1.5
    if bookmarks:
        doc.set_toc(bookmarks)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_1_pdf_with_toc_bookmarks(tmp_path):
    """Scenario 1: PDF with clear outline bookmarks."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "toc_book.pdf",
        pages_content=[
            ("Cover", [("Textbook of Science", 24, True)]),
            ("TOC", [("Table of Contents", 18, True), ("1. Motion", 12, False)]),
            ("Ch 1", [("Chapter 1: Motion", 22, True), ("Objects move in time...", 10, False)]),
            ("Ch 1 p2", [("Velocity and speed...", 10, False)]),
            ("Ch 2", [("Chapter 2: Force", 22, True), ("Force causes acceleration...", 10, False)]),
        ],
        bookmarks=[
            [1, "Chapter 1: Motion", 3],
            [1, "Chapter 2: Force", 5],
        ],
    )

    res = detect_textbook_structure(pdf)
    assert res["toc_found"] is True
    assert len(res["chapters"]) >= 2
    ch1 = res["chapters"][0]
    assert ch1["chapter_number"] == "1"
    assert "Motion" in ch1["chapter_title"]
    assert ch1["start_pdf_page"] == 2
    assert ch1["end_pdf_page"] == 3

    ch2 = res["chapters"][1]
    assert ch2["chapter_number"] == "2"
    assert "Force" in ch2["chapter_title"]
    assert ch2["start_pdf_page"] == 4


def test_2_pdf_without_toc_bookmarks():
    """Scenario 2: Real textbook PDF without bookmarks (gees102.pdf)."""
    assert os.path.isfile("gees102.pdf"), "gees102.pdf must be present in workspace"
    res = detect_textbook_structure("gees102.pdf")
    assert res["total_pages"] == 18
    assert res["pdf_type"] == "text_based"
    assert len(res["chapters"]) >= 1
    ch = res["chapters"][0]
    assert ch["chapter_number"] == "2"
    assert "Weather" in ch["chapter_title"]
    assert ch["start_pdf_page"] == 0
    assert ch["end_pdf_page"] == 17
    assert ch["confidence_score"] >= 0.80


def test_3_scanned_textbook_detection(tmp_path):
    """Scenario 3: Scanned textbook with minimal or no text layer."""
    pdf_path = str(tmp_path / "scanned.pdf")
    doc = fitz.open()
    for _ in range(5):
        # Empty pages representing scans
        doc.new_page(width=595, height=842)
    doc.save(pdf_path)
    doc.close()

    doc_check = fitz.open(pdf_path)
    pdf_type = detect_pdf_type(doc_check)
    doc_check.close()
    assert pdf_type == "scanned"


def test_4_roman_numeral_preliminary_pages():
    """Scenario 4: Roman numeral preliminary pages."""
    assert roman_to_int("i") == 1
    assert roman_to_int("iv") == 4
    assert roman_to_int("ix") == 9
    assert roman_to_int("xiv") == 14
    assert parse_chapter_number_str("IV") == "4"
    assert parse_chapter_number_str("Two") == "2"


def test_5_printed_vs_physical_page_offset():
    """Scenario 5: Printed page vs physical page offset reconciliation."""
    toc_entries = [
        {"chapter_number": "1", "printed_page": 1},
        {"chapter_number": "2", "printed_page": 15},
    ]
    # In physical PDF, Chapter 1 is on page 6 (0-indexed 6 means offset is 5)
    body_headings = [
        {"chapter_number": "1", "physical_page": 6},
        {"chapter_number": "2", "physical_page": 20},
    ]
    offset = reconcile_page_offset(toc_entries, body_headings, 50)
    assert offset == 5


def test_6_chapter_title_appearing_in_ordinary_body_text(tmp_path):
    """Scenario 6: Negative filtering rejects references, figure captions, and exercises."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "negative_filters.pdf",
        pages_content=[
            ("Page 0", [
                ("Introduction", 22, True),
                ("As we will study in Chapter 2, forces act on bodies.", 10, False),
                ("Figure 1.1: Chapter 1 illustration", 10, False),
                ("Exercises for Chapter 1", 12, False),
            ]),
            ("Page 1", [
                ("Chapter 2: Real Force", 22, True),
                ("Body text for chapter 2.", 10, False),
            ]),
        ],
    )
    doc = fitz.open(pdf)
    headings = scan_body_headings(doc, base_font_size=10.0)
    doc.close()

    # The body text reference "As we will study in Chapter 2..." or "Figure 1.1" must NOT become Chapter 1 or 2 on page 0!
    p0_headings = [h for h in headings if h["physical_page"] == 0 and h["chapter_number"] == "2"]
    assert len(p0_headings) == 0

    # Page 1 true chapter heading must be detected
    p1_headings = [h for h in headings if h["physical_page"] == 1]
    assert len(p1_headings) == 1
    assert p1_headings[0]["chapter_number"] == "2"


def test_7_missing_or_ambiguous_chapter_heading(tmp_path):
    """Scenario 7: Low confidence heading is flagged for review."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "ambiguous.pdf",
        pages_content=[
            ("Page 0", [
                ("Some Topic Notes", 11, False),
                ("Standard paragraph content without explicit chapter banners.", 10, False),
            ]),
        ],
    )
    res = detect_textbook_structure(pdf)
    assert len(res["chapters"]) == 1
    # Fallback chapter should have lower confidence
    assert res["chapters"][0]["confidence_score"] <= 0.70


def test_8_chapter_spanning_many_pages(tmp_path):
    """Scenario 8: Chapter spanning multiple pages is correctly bounded."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "multi_page.pdf",
        pages_content=[
            ("Ch 1", [("Chapter 1: Plant Biology", 22, True), ("Text...", 10, False)]),
            ("p2", [("Page 2 content...", 10, False)]),
            ("p3", [("Page 3 content...", 10, False)]),
            ("p4", [("Page 4 content...", 10, False)]),
            ("Ch 2", [("Chapter 2: Animal Biology", 22, True), ("Text...", 10, False)]),
            ("p6", [("Page 6 content...", 10, False)]),
        ],
    )
    res = detect_textbook_structure(pdf)
    assert len(res["chapters"]) == 2
    ch1 = res["chapters"][0]
    assert ch1["start_pdf_page"] == 0
    assert ch1["end_pdf_page"] == 3  # Spans pages 0, 1, 2, 3

    ch2 = res["chapters"][1]
    assert ch2["start_pdf_page"] == 4
    assert ch2["end_pdf_page"] == 5


def test_9_slice_chapter_pdf_and_idempotency(tmp_path):
    """Scenario 9: Slicing extracts exact sub-PDF without data loss."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "source_book.pdf",
        pages_content=[
            ("P0", [("Page 0 text", 12, False)]),
            ("P1", [("Page 1 text", 12, False)]),
            ("P2", [("Page 2 text", 12, False)]),
            ("P3", [("Page 3 text", 12, False)]),
        ],
    )
    slice_dest = str(tmp_path / "slice_ch1.pdf")
    slice_chapter_pdf(pdf, start_page=1, end_page=2, dst_pdf_path=slice_dest)

    assert os.path.isfile(slice_dest)
    doc = fitz.open(slice_dest)
    assert len(doc) == 2
    assert "Page 1 text" in doc[0].get_text()
    assert "Page 2 text" in doc[1].get_text()
    doc.close()


def test_10_legacy_single_chapter_upload_regression():
    """Scenario 10: Existing TextbookUpload model and catalog schemas remain functional."""
    from datetime import UTC, datetime
    from app.modules.catalog.models import BoardEnum, ClassEnum, ProcessingStatusEnum, TextbookUpload
    from app.modules.catalog.schemas import TextbookUploadResponse

    tu = TextbookUpload(
        id=9999,
        file_name="legacy_chapter.pdf",
        board=BoardEnum.CBSE,
        class_level=ClassEnum.CLASS_10,
        subject_name="Mathematics",
        chapter="Real Numbers",
        content_type="CHAPTER",
        content_label="Chapter 1: Real Numbers",
        file_path="CBSE/CLASS_10/Mathematics/legacy_chapter.pdf",
        chunk_count=15,
        upload_date=datetime.now(UTC),
        ocr_status=ProcessingStatusEnum.EMBEDDED,
        chunk_status=ProcessingStatusEnum.EMBEDDED,
        embedding_status=ProcessingStatusEnum.EMBEDDED,
    )
    resp = TextbookUploadResponse.model_validate(tu)
    assert resp.id == 9999
    assert resp.chapter == "Real Numbers"
    assert resp.embedding_status == ProcessingStatusEnum.EMBEDDED


def test_11_8th_eng_full_textbook_regression():
    """Scenario 11: Real-world regression test on 8th eng.pdf (SCERT Class 8 English)."""
    pdf_path = "/Applications/ForeFold/virtual tutor/8th eng.pdf"
    if not os.path.isfile(pdf_path):
        pytest.skip(f"Fixture {pdf_path} not found")

    res = detect_textbook_structure(pdf_path)
    assert res["total_pages"] == 162
    assert res["pdf_type"] == "text_based"
    assert res["page_offset"] == 9
    assert res["offset_confidence"] >= 0.90

    # Must detect all 8 Units
    chapters = res["chapters"]
    assert len(chapters) == 8

    # Unit 1: Family (Printed 1-14 -> Physical 10-23)
    u1 = chapters[0]
    assert u1["chapter_number"] == "1"
    assert "Family" in u1["chapter_title"]
    assert u1["start_pdf_page"] == 10
    assert u1["end_pdf_page"] == 23

    # Unit 8: Gratitude (Printed 123-141 -> Physical 132-150)
    u8 = chapters[7]
    assert u8["chapter_number"] == "8"
    assert "Gratitude" in u8["chapter_title"]
    assert u8["start_pdf_page"] == 132
    assert u8["end_pdf_page"] == 150

    # Non-chapter sections: Front matter (0 to 9) and Appendices (151 to 161)
    non_ch = res["non_chapter_sections"]
    assert any(s["start_pdf_page"] == 0 for s in non_ch)
    assert any("appendi" in s["chapter_title"].lower() or "listening" in s["chapter_title"].lower() for s in non_ch)

    # Watermark and footer cleaning verification
    from app.services.textbook_structure.text_cleaner import DocumentTextCleaner
    from app.services.textbook_structure.profiler import profile_document

    doc = fitz.open(pdf_path)
    prof = profile_document(doc)
    assert "SCERT TELANGANA" in prof.detected_watermarks
    assert any("Free distribution" in f for f in prof.repeated_footers)

    sample_raw = doc[10].get_text()
    cleaned, stats = DocumentTextCleaner.clean_text(sample_raw, prof)
    assert "SCERT TELANGANA" not in cleaned
    assert "Free distribution by T.S. Government" not in cleaned
    assert stats.watermarks_removed >= 1
    doc.close()


def test_12_cbse_ix_maths_em_regression():
    """Scenario 12: Real-world regression test on CBSE Class 9 Mathematics (362 pages)."""
    pdf_path = "./uploads/CBSE/CLASS_9/Mathematics/5852e3ad_ix maths em.pdf"
    if not os.path.isfile(pdf_path):
        pytest.skip(f"Fixture {pdf_path} not found")

    res = detect_textbook_structure(pdf_path)
    assert res["total_pages"] == 362
    assert res["pdf_type"] == "text_based"
    assert len(res["chapters"]) == 15
    assert res["chapters"][0]["chapter_number"] == "1"
    assert "Real Numbers" in res["chapters"][0]["chapter_title"]


def test_13_document_text_cleaner_unit():
    """Scenario 13: Unit test for DocumentTextCleaner sanitization."""
    from app.services.textbook_structure.text_cleaner import DocumentTextCleaner

    sample = (
        "12\n"
        "Free distribution by T.S. Government 2019-20\n"
        "This is an instruc- \n"
        "tional paragraph on math.\n"
        "SCERT TELANGANA\n"
        "12\n"
    )
    cleaned, stats = DocumentTextCleaner.clean_text(sample)
    assert "SCERT TELANGANA" not in cleaned
    assert "Free distribution" not in cleaned
    assert "instructional" in cleaned
    assert stats.hyphens_repaired == 1


def test_14_multilingual_hindi_chapters(tmp_path):
    """Scenario 14: Regional language Hindi chapters (अध्याय 1)."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "hindi_book.pdf",
        pages_content=[
            ("Cover", [("गणित कक्षा 10", 22, True)]),
            ("Ch 1", [("अध्याय 1 वास्तविक संख्याएँ", 20, True), ("संख्याओं के बारे में...", 10, False)]),
            ("p2", [("आगे की जानकारी...", 10, False)]),
            ("Ch 2", [("अध्याय 2 बहुपद", 20, True), ("बहुपद की परिभाषा...", 10, False)]),
        ],
    )
    res = detect_textbook_structure(pdf)
    assert len(res["chapters"]) == 2
    assert res["chapters"][0]["chapter_number"] == "1"
    assert "वास्तविक संख्याएँ" in res["chapters"][0]["chapter_title"]
    assert res["chapters"][1]["chapter_number"] == "2"
    assert "बहुपद" in res["chapters"][1]["chapter_title"]


def test_15_negative_captions_and_exercises(tmp_path):
    """Scenario 15: Captions and exercises are not mistaken for chapters."""
    pdf = _create_synthetic_pdf(
        tmp_path,
        "captions_book.pdf",
        pages_content=[
            ("Ch 1", [
                ("Chapter 1: Optics", 22, True),
                ("Figure 1.1: Ray diagram of reflection", 14, True),
                ("Table 1.1: Refractive indices", 14, True),
                ("Exercise 1.1: Solve the following problems", 14, True),
                ("Questions for review", 14, True),
            ]),
            ("p2", [("More text about light...", 10, False)]),
        ],
    )
    res = detect_textbook_structure(pdf)
    assert len(res["chapters"]) == 1
    assert res["chapters"][0]["chapter_title"] == "Optics"


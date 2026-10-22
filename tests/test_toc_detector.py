import fitz

from app.services.textbook_structure import parse_pdf_outline
from app.services.textbook_structure.toc_detector import UniversalTocDetector


def _table_toc_doc() -> fitz.Document:
    """SCERT-style contents: S.No | Name | Periods | Month | Page, one cell per text line."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((240, 118), "CONTENTS")
    rows = [("1.", "Animals - The Base of our life", "15", "June", "1"),
            ("2.", "Agriculture - Crops", "8", "June", "15"),
            ("3.", "March of the Market", "10", "July", "26")]
    for i, (n, title, periods, month, pg) in enumerate(rows):
        y = 180 + 25 * i
        for x, txt in ((89, n), (141, title), (350, periods), (410, month), (490, pg)):
            page.insert_text((x, y), txt)
    return doc


def test_row_table_toc():
    entries = UniversalTocDetector.parse_toc(_table_toc_doc())
    assert [(e.chapter_number, e.chapter_title, e.printed_start_page) for e in entries] == [
        ("1", "Animals - The Base of our life", 1),
        ("2", "Agriculture - Crops", 15),
        ("3", "March of the Market", 26),
    ]


def _tabular_row_doc(content: str) -> fitz.Document:
    """One tabular TOC row: number+page-range block on the left, title/content words across the page."""
    doc = fitz.open()
    page = doc.new_page()  # 595 wide
    page.insert_text((60, 120), "CONTENTS")
    page.insert_text((60, 200), "1.\n1-15")  # number + page range in one block
    page.insert_text((80, 200), "Knowing")  # title column (0.12-0.31 W)
    page.insert_text((190, 200), content)  # spills into the 'readings' column
    return doc


def test_tabular_title_overflow_is_not_a_reading():
    (e,) = UniversalTocDetector._parse_tabular_toc(_tabular_row_doc("Our Numbers   June")[0], 0)
    assert (e.chapter_title, e.hierarchy_level, e.sub_readings) == ("Knowing Our Numbers", "chapter", [])


def test_tabular_marked_readings_are_kept():
    (e,) = UniversalTocDetector._parse_tabular_toc(_tabular_row_doc("A. The Blanket B. My Mother")[0], 0)
    assert e.hierarchy_level == "unit" and e.sub_readings == ["A. The Blanket", "B. My Mother"]


def test_index_table_with_page_before_periods():
    """SCERT Science layout: heading 'INDEX' drawn after the table, columns S.No | Name | Page | Periods | Month."""
    doc = fitz.open()
    for _ in range(9):
        doc.new_page()
    page = doc.new_page()
    rows = [("1.", "Our food", "1", "10", "June"), ("2", "Playing with magnets", "11", "12", "June"),
            ("3", "Rain: where does it come from?", "21", "10", "July"), ("4.", "What do animals eat?", "28", "12", "July"),
            ("5", "Materials - Objects", "41", "12", "August"), ("6", "Habitat", "50", "12", "August")]
    for i, (n, title, pg, periods, month) in enumerate(rows):
        y = 180 + 25 * i
        for x, txt in ((95, n), (137, title), (375, pg), (434, periods), (488, month)):
            page.insert_text((x, y), txt)
    page.insert_text((143, 110), "INDEX")  # inserted last: comes after the rows in extraction order

    assert UniversalTocDetector.find_toc_pages(doc) == [9]
    entries = UniversalTocDetector.parse_toc(doc)
    assert [(e.chapter_title, e.printed_start_page) for e in entries][:3] == [
        ("Our food", 1), ("Playing with magnets", 11), ("Rain: where does it come from?", 21)
    ]
    assert [e.printed_start_page for e in entries][-1] == 50


def test_merged_pdf_outline_keeps_only_real_chapters():
    doc = fitz.open()
    for i in range(40):
        page = doc.new_page()
        if i in (10, 20):
            page.insert_text((60, 80), "Our Food" if i == 10 else "Playing with Magnets", fontsize=28)
            page.insert_text((60, 140), "Body text of the chapter.", fontsize=11)
    doc.set_toc([
        [1, "initialpages.pdf", 3],
        [1, "01_Mana Aaharam-EM.pdf", 11],
        [1, "02_Ayaskantalato Aatalu - 2022 EM.pdf", 21],
        [1, "03_Varsham - 2022 - Final - EM.pdf", 31],
        [1, "14_Jantuvullalo Chalanam - 2022 - Final -EM.pdf", 1],  # broken link
        [1, "VI General Science EM 2026-27 - Coverpage.pdf", 1],
        [2, "01_Mana Aaharam-EM.pdf", 1],  # merge-tool duplicates
        [3, "Page 1", 1],
    ])
    entries = parse_pdf_outline(doc)
    assert [(e.chapter_number, e.chapter_title, e.printed_start_page) for e in entries] == [
        ("1", "Our Food", 10),
        ("2", "Playing with Magnets", 20),
        ("3", "Varsham", 30),  # no heading on that page: cleaned file name
    ]
    assert all(e.confidence < 0.85 for e in entries)


def _reading_label_doc() -> fitz.Document:
    """SSC English layout: 'N. A. Reading <title>' / 'B. Reading (Poem) <title> <month> <range>' / 'C. Reading <title>'."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((250, 72), "CONTENTS")
    rows = [
        ((75, "Readiness Programme"), (197, "Bridging gap activity"), (414, "June"), (495, "1-3")),
        ((75, "1."), (97, "A Reading"), (205, "Peace and Harmony")),
        ((97, "B. Reading (Poem)"), (205, "I Want Peace"), (406, "June-July"), (489, "4-18")),
        ((97, "C. Reading"), (205, "Grand Contest in the Forest")),
        ((75, "2."), (97, "A. Reading"), (205, "Telangana, The Pride of the People")),
        ((97, "B. Reading (Poem)"), (205, "In the Bazaars of Hyderabad"), (406, "July"), (489, "19-30"), (530, "TELANGANA")),
        ((97, "C. Reading"), (205, "Wilma Rudolph")),
        ((97, "Appendix-I"), (205, "Texts for Listening"), (489, "31-36")),
    ]
    for i, cells in enumerate(rows):
        for x, txt in cells:
            page.insert_text((x, 120 + 22 * i), txt)
    return doc


def test_reading_label_units():
    entries = UniversalTocDetector.parse_toc(_reading_label_doc())
    got = [(e.chapter_number, e.hierarchy_level, e.chapter_title, e.printed_start_page, e.printed_end_page) for e in entries]
    assert got == [
        ("0", "chapter", "Bridging gap activity", 1, 3),
        ("1", "unit", "Unit 1: Peace and Harmony", 4, 18),
        ("2", "unit", "Unit 2: Telangana, The Pride of the People", 19, 30),
        ("App", "appendix", "Appendix-I Texts for Listening", 31, 36),
    ]
    assert entries[1].sub_readings == ["A. Peace and Harmony", "B. I Want Peace (Poem)", "C. Grand Contest in the Forest"]
    assert entries[2].sub_readings[-1] == "C. Wilma Rudolph"


def test_parse_toc_keeps_best_ordered_result(monkeypatch):
    from app.services.textbook_structure.toc_detector import TocEntry

    def entry(pg):
        return TocEntry("1", "x", "chapter", pg, None, 0)

    doc = fitz.open()
    doc.new_page()
    U = UniversalTocDetector
    monkeypatch.setattr(U, "_parse_reading_rows_toc", classmethod(lambda cls, p, n: []))
    monkeypatch.setattr(U, "_parse_tabular_toc", classmethod(lambda cls, p, n: [entry(5), entry(3)]))  # out of order
    monkeypatch.setattr(U, "_parse_line_toc", classmethod(lambda cls, p, n: [entry(1), entry(9), entry(20)]))
    monkeypatch.setattr(U, "_parse_row_toc", classmethod(lambda cls, p, n: [entry(1)]))
    assert [e.printed_start_page for e in U.parse_toc(doc, [0])] == [1, 9, 20]

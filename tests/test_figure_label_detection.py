import fitz

from app.services.image_service.textbook_image_extraction import classify_image_type
from app.services.pdf_extract_pipeline.embedded_figure_refinement import _stitch_tiles
from app.services.pdf_extract_pipeline.figure_pairing import _caption_like, _is_inline_figure_reference


def _page_with_lines(lines: list[str]) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    return page


def _label_hits(page: fitz.Page, needle: str) -> list[bool]:
    return [_is_inline_figure_reference(page, r) for r in page.search_for(needle)]


def test_caption_label_is_not_a_reference():
    page = _page_with_lines(["Fig. 12.4: Big Dipper, Little Dipper and Pole Star"])
    assert _label_hits(page, "Fig. 12.4") == [False]


def test_prose_mentions_are_references():
    for line, label in (
        ("Star. You may refer to Fig. 12.4 for help.", "Fig. 12.4"),
        ("most of these craters (Fig. 12.9). Most of them", "Fig. 12.9"),
        ("Fig. 12.2 shows bright stars in one part of the sky.", "Fig. 12.2"),
    ):
        page = _page_with_lines([line])
        assert _label_hits(page, label) == [True], line


def test_reference_wrapped_to_next_line():
    page = _page_with_lines(["An artist's representation of the Solar System is given in", "Fig. 12.12. Is the order correct?"])
    assert _label_hits(page, "Fig. 12.12") == [True]


def test_second_caption_on_same_line():
    page = _page_with_lines(["Fig. 2.4. The Qutub Minar. Fig. 2.5. A section of the mosque"])
    assert _label_hits(page, "Fig. 2.5") == [False]


def test_subpanel_marker_above_caption():
    page = _page_with_lines(["(a)", "Fig. 12.9: The Moon"])
    assert _label_hits(page, "Fig. 12.9") == [False]


def test_map_tiles_stitch_into_one_rect():
    tiles = [
        fitz.Rect(52, 76, 457, 482),
        fitz.Rect(457, 76, 528, 482),
        fitz.Rect(52, 482, 457, 552),
        fitz.Rect(457, 482, 528, 552),
        fitz.Rect(60, 600, 200, 700),
    ]
    out = _stitch_tiles(tiles)
    assert fitz.Rect(52, 76, 528, 552) in out
    assert len(out) == 2


def test_caption_like_rejects_prose_and_footers():
    assert _caption_like("Indian Astronomical Observatory, Hanle")
    assert not _caption_like("In Indian astronomy, the term nakshatra is used to denote")
    assert not _caption_like("patterns?")
    assert not _caption_like("Reprint 2026-27")


def test_image_type_ignores_chapter_theme_text():
    ctx = "Chapter 2 - Reshaping India's Political Map. The region and its territory"
    assert classify_image_type("Fig. 2.7. A 19th-century painting of the sultan", ctx) == "historical_photo"
    assert classify_image_type("Fig. 2.2.", "", "type: timeline; subject: events 1206-1498") == "chart"
    assert classify_image_type("Fig. 2.12. Kingdoms in the Deccan", ctx, "type: map; subject: kingdoms") == "map"

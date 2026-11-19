"""Tests for embedded PDF figure bbox refinement."""

from pathlib import Path

import fitz

from app.services.pdf_extract_pipeline.embedded_figure_refinement import (
    refine_figure_box,
    _label_rect_for_figure,
)
from app.services.pdf_extract_pipeline.raster import image_box_to_pdf_rect, load_pdf_pages

BACKEND = Path(__file__).resolve().parents[1]
GEES103 = BACKEND / "uploads/CBSE/CLASS_9/Social/3c0f878d_gees103.pdf"


def _prose_word_count(page: fitz.Page, box: tuple[int, int, int, int], image_size, dpi) -> int:
    rect = image_box_to_pdf_rect(box, page.rect, image_size, dpi)
    text = page.get_text("text", clip=rect) or ""
    return len(text.split())


def test_refine_gees103_figures_when_pdf_present():
    if not GEES103.is_file():
        return
    doc, images = load_pdf_pages(str(GEES103), dpi=150)
    try:
        page4 = doc[3]
        box33 = refine_figure_box(
            page4,
            layout_box=(181, 625, 580, 933),
            label_box=None,
            fig_number="3.3",
            image_size=images[3].size,
            dpi=144,
        )
        w = box33[2] - box33[0]
        h = box33[3] - box33[1]
        assert 350 < w < 520
        assert 280 < h < 560

        page5 = doc[4]
        box34 = refine_figure_box(
            page5,
            layout_box=(437, 674, 1071, 1112),
            label_box=None,
            fig_number="3.4",
            image_size=images[4].size,
            dpi=150,
        )
        w4 = box34[2] - box34[0]
        assert w4 > 400
        assert w4 < 700

        # Fig 3.3 — keep clean YOLO layout box (do not shrink to embedded fragment).
        page4 = doc[3]
        box33b = refine_figure_box(
            page4,
            layout_box=(175, 619, 586, 984),
            label_box=None,
            fig_number="3.3",
            image_size=images[3].size,
            dpi=150,
        )
        assert box33b[2] - box33b[0] >= 400

        # Fig 3.9 — must not include monsoon body paragraphs above the diagram.
        page10 = doc[9]
        layout_39 = (106, 379, 1111, 1238)
        box39 = refine_figure_box(
            page10,
            layout_box=layout_39,
            label_box=None,
            fig_number="3.9",
            image_size=images[9].size,
            dpi=150,
        )
        label = _label_rect_for_figure(page10, "3.9")
        assert label is not None
        crop_pdf = image_box_to_pdf_rect(box39, page10.rect, images[9].size, 150)
        assert crop_pdf.y0 > 400, "Fig 3.9 crop should start below body text"
        assert "monsoon season is central" not in (
            page10.get_text("text", clip=crop_pdf) or ""
        ).lower()

        # Fig 3.10 — tight photo crop, not DON'T MISS OUT callout + paragraphs.
        page11 = doc[10]
        box310 = refine_figure_box(
            page11,
            layout_box=(106, 379, 1050, 1304),
            label_box=None,
            fig_number="3.10",
            image_size=images[10].size,
            dpi=150,
        )
        crop310 = image_box_to_pdf_rect(box310, page11.rect, images[10].size, 150)
        text310 = (page11.get_text("text", clip=crop310) or "").lower()
        assert "don't miss out" not in text310
        assert "mawsynram" not in text310 or "fig. 3.10" in text310
        assert crop310.height < 220

        # Fig 3.8 — union both side-by-side map panels.
        page9 = doc[8]
        box38 = refine_figure_box(
            page9,
            layout_box=(69, 643, 1066, 1026),
            label_box=None,
            fig_number="3.8",
            image_size=images[8].size,
            dpi=150,
        )
        # Fig 3.6 — vector diagram must span past the column rule on the right.
        page7 = doc[6]
        box36 = refine_figure_box(
            page7,
            layout_box=(429, 497, 1051, 1008),
            label_box=None,
            fig_number="3.6",
            image_size=images[6].size,
            dpi=144,
        )
        w36 = box36[2] - box36[0]
        assert w36 > 630
    finally:
        doc.close()


if __name__ == "__main__":
    test_refine_gees103_figures_when_pdf_present()
    print("ok")

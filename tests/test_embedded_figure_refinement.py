"""Tests for embedded PDF figure bbox refinement."""

from pathlib import Path

from app.services.pdf_extract_pipeline.embedded_figure_refinement import refine_figure_box
from app.services.pdf_extract_pipeline.raster import load_pdf_pages

BACKEND = Path(__file__).resolve().parents[1]
GEES103 = BACKEND / "uploads/CBSE/CLASS_9/Social/3c0f878d_gees103.pdf"


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
            dpi=150,
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
        assert w4 > 600
    finally:
        doc.close()


if __name__ == "__main__":
    test_refine_gees103_figures_when_pdf_present()
    print("ok")

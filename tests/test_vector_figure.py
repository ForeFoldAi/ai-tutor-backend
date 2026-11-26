"""Vector-diagram figure location, printed captions and watermark stripping (synthetic PDFs)."""

from io import BytesIO

import fitz
from PIL import Image

from app.services.pdf_extract_pipeline.vector_figure import (
    body_font_size,
    is_figure_label_text,
    is_prose_line,
    locate_figure,
    page_text_lines,
    printed_caption,
    prose_mask_rects,
    render_region,
    strip_recurring_images,
)

BODY = "Let us now prove the theorem about the triangle shown in the figure given here carefully."


def _triangle_page(doc: fitz.Document, *, label: str = "Fig. 6.14", body_right: bool = False) -> fitz.Page:
    page = doc.new_page(width=468, height=684)
    # body paragraph above the figure
    for i in range(3):
        page.insert_text((40, 90 + i * 14), BODY, fontsize=11)
    # triangle diagram
    x0 = 100 if body_right else 180
    # each side its own path, as in real textbook PDFs
    for a, b in (((x0, 300), (x0 + 120, 300)), ((x0 + 120, 300), (x0 + 60, 200)),
                 ((x0 + 60, 200), (x0, 300)), ((x0 - 20, 260), (x0 + 140, 260))):
        page.draw_line(a, b, color=(0, 0.6, 0.9), width=1.5)
    if body_right:
        for i in range(4):
            page.insert_text((270, 215 + i * 14), "and the sides are in the same ratio", fontsize=11)
    page.insert_text((x0 + 35, 325), label, fontsize=11)
    page.insert_text((40, 370), BODY, fontsize=11)
    return page


def _label_rect(page: fitz.Page, text: str) -> fitz.Rect:
    return page.search_for(text)[0]


def test_label_and_prose_classification():
    assert is_figure_label_text("Fig. 6.14") == "6.14"
    assert is_figure_label_text("Figure 9.1 Schematic representation") == "9.1"
    assert is_figure_label_text("see the figure") is None
    assert is_prose_line(BODY)
    assert not is_prose_line("Hypotenuse")
    assert not is_prose_line("Fig. 6.14")


def test_locate_vector_triangle_above_label():
    doc = fitz.open()
    page = _triangle_page(doc)
    fig = locate_figure(page, _label_rect(page, "Fig. 6.14"))
    assert fig is not None
    assert fig.vector_paths >= 2
    assert fig.vector_paths >= 4
    assert fig.core.contains(fitz.Rect(160, 200, 320, 300))  # includes the horizontal line
    # excludes the paragraphs above and below and the label itself
    assert fig.rect.y0 > 100 and fig.rect.y1 < 320


def test_prose_beside_figure_is_excluded_and_masked():
    doc = fitz.open()
    page = _triangle_page(doc, body_right=True)
    # a body-size fragment right next to the drawing, e.g. "(Theorem 6.1)"
    page.insert_text((200, 201), "(Theorem 6.1)", fontsize=11)
    fig = locate_figure(page, _label_rect(page, "Fig. 6.14"))
    assert fig is not None
    assert fig.rect.x1 < 265  # paragraph to the right is not part of the figure
    mask = prose_mask_rects(page, fig)
    theorem = _label_rect(page, "(Theorem 6.1)")
    assert any(m.intersects(theorem) for m in mask)
    png = render_region(page, fig.rect, mask=mask)
    img = Image.open(BytesIO(png))
    assert 0 < img.width <= fig.rect.width * 2.5 + 2


def test_no_drawing_returns_none():
    doc = fitz.open()
    page = doc.new_page(width=468, height=684)
    page.insert_text((40, 300), BODY, fontsize=11)
    page.insert_text((200, 330), "Fig. 1.1", fontsize=11)
    assert locate_figure(page, _label_rect(page, "Fig. 1.1")) is None


def test_printed_caption_reads_continuation_and_stops_at_body():
    doc = fitz.open()
    page = doc.new_page(width=468, height=684)
    for i in range(3):
        page.insert_text((40, 80 + i * 14), BODY, fontsize=11)
    page.insert_text((40, 300), "Fig. 5 - Chauri Chaura, 1922.", fontsize=8.5)
    page.insert_text((40, 311), "A peaceful demonstration turned into a violent clash.", fontsize=8.5)
    page.insert_text((40, 340), BODY, fontsize=11)
    cap = printed_caption(page, _label_rect(page, "Fig. 5"))
    assert cap == "Fig. 5 - Chauri Chaura, 1922. A peaceful demonstration turned into a violent clash."
    assert body_font_size(page_text_lines(page)) == 11


def test_printed_caption_dedupes_overprinted_label():
    doc = fitz.open()
    page = doc.new_page(width=468, height=684)
    for i in range(3):
        page.insert_text((40, 80 + i * 14), BODY, fontsize=11)
    for _ in range(3):
        page.insert_text((40, 300), "Figure 9.15", fontsize=9)
    page.insert_text((40, 311), "Ray diagrams for a concave lens", fontsize=9)
    cap = printed_caption(page, _label_rect(page, "Figure 9.15"))
    assert cap.count("Figure 9.15") == 1
    assert cap.endswith("Ray diagrams for a concave lens")


def _png(color: str, size=(40, 40)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_strip_recurring_images_removes_watermark_only():
    doc = fitz.open()
    stamp = _png("gray")
    photo = _png("red", (60, 50))
    for i in range(5):
        page = doc.new_page(width=468, height=684)
        page.insert_image(fitz.Rect(30, 170, 430, 570), stream=stamp)
        if i == 2:
            page.insert_image(fitz.Rect(100, 100, 220, 200), stream=photo)
    clean, removed = strip_recurring_images(doc)
    assert removed >= 1  # a shared image xref is removed once for every page
    real = [i for p in clean for i in p.get_image_info() if int(i.get("width") or 0) > 2]
    assert len(real) == 1 and fitz.Rect(real[0]["bbox"]).width > 100
    # source document untouched
    assert sum(len(p.get_image_info()) for p in doc) == 6


def test_locate_figure_terminates_with_degenerate_cluster(monkeypatch):
    """An empty cluster is never 'contained' and adds nothing on union; the
    multi-panel merge loop used to spin forever on it (Class 6 'Lines and Angles')."""
    import fitz

    from app.services.pdf_extract_pipeline import vector_figure as vf

    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    core = fitz.Rect(100, 100, 300, 300)
    empty = fitz.Rect(305, 150, 305, 150)  # zero-size, sits beside the core
    monkeypatch.setattr(vf, "figure_elements", lambda *a, **k: ([core, empty], 2))
    monkeypatch.setattr(vf, "cluster_rects", lambda rects, gap=0: [core, empty])
    monkeypatch.setattr(vf, "_contains_prose", lambda *a, **k: False)
    fig = vf.locate_figure(page, fitz.Rect(150, 310, 250, 322), lines=[])
    assert fig is not None
    assert fig.core == core

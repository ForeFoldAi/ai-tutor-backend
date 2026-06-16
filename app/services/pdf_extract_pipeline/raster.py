"""PDF page rasterization for full-page ML analysis."""

from __future__ import annotations

import fitz
from PIL import Image


def render_page(page: fitz.Page, dpi: int = 144) -> Image.Image:
    """Rasterize one PDF page to RGB PIL image."""
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    if pix.width > 3000 or pix.height > 3000:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def load_pdf_pages(
    pdf_path: str,
    *,
    dpi: int = 144,
    max_pages: int | None = None,
) -> tuple[fitz.Document, list[Image.Image]]:
    doc = fitz.open(pdf_path)
    n = len(doc)
    if max_pages is not None:
        n = min(n, max_pages)
    images = [render_page(doc[i], dpi) for i in range(n)]
    return doc, images


def image_box_to_pdf_rect(
    box: tuple[int, int, int, int],
    page_rect: fitz.Rect,
    image_size: tuple[int, int],
    dpi: int,
) -> fitz.Rect:
    scale = dpi / 72.0
    xmin, ymin, xmax, ymax = box
    rect = fitz.Rect(
        xmin / scale,
        ymin / scale,
        xmax / scale,
        ymax / scale,
    )
    return rect & page_rect


def pdf_rect_to_image_box(rect: fitz.Rect, dpi: int) -> tuple[int, int, int, int]:
    scale = dpi / 72.0
    return (
        int(rect.x0 * scale),
        int(rect.y0 * scale),
        int(rect.x1 * scale),
        int(rect.y1 * scale),
    )

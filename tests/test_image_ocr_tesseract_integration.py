"""Real Tesseract on synthetic Pillow images. Skipped when the tesseract binary is not installed."""

from __future__ import annotations

import shutil
from io import BytesIO

import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.config import TESSERACT_CMD
from app.modules.image_understanding.ocr.tesseract import TesseractOCRService
from app.modules.image_understanding.quality import assess_quality
from app.modules.image_understanding.routing import Task, decide

pytestmark = pytest.mark.skipif(
    not (TESSERACT_CMD or shutil.which("tesseract")),
    reason="tesseract binary not installed (brew install tesseract / Docker image)",
)

_LINES = [
    "Chapter 3: The Water Cycle",
    "Water evaporates from rivers, lakes and oceans.",
    "The vapour rises, cools and condenses into clouds.",
    "Clouds release water back to the land as rain.",
    "Rain water flows into rivers and seeps into the ground.",
    "Q1. Why is the water cycle important for living things?",
]


def _png(im: Image.Image) -> bytes:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _printed_page() -> Image.Image:
    im = Image.new("RGB", (1400, 760), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(_LINES):
        d.text((40, 40 + i * 115), line, fill="black", font=font)
    return im


def _graph() -> Image.Image:
    im = Image.new("RGB", (1000, 800), "white")
    d = ImageDraw.Draw(im)
    d.line((100, 700, 900, 700), fill="black", width=3)
    d.line((100, 700, 100, 100), fill="black", width=3)
    d.line((100, 700, 850, 150), fill="blue", width=4)
    d.text((300, 40), "Find the slope", fill="black", font=ImageFont.load_default(size=40))
    return im


@pytest.fixture(scope="module")
def svc() -> TesseractOCRService:
    return TesseractOCRService()


def test_tesseract_version_and_language(svc):
    st = svc.status()
    assert st["available"] and st["version"].startswith("5.")
    assert st["lang_available"], f"{svc.lang} traineddata missing"


def test_clean_printed_text_routes_to_tesseract(svc):
    data = _png(_printed_page())
    ocr = svc.run(data)
    assert ocr.error is None and ocr.has_text
    assert "evaporates" in ocr.text and "water cycle" in ocr.text.lower()
    assert ocr.words and all(w.width > 0 and w.height > 0 for w in ocr.words)
    d = decide(ocr, assess_quality(data), "explain this")
    assert d.task is Task.TEXT_ONLY, (d.reason, ocr.average_confidence, ocr.text_coverage)


def test_blank_image_has_no_text(svc):
    data = _png(Image.new("RGB", (800, 600), "white"))
    assert not svc.run(data).has_text
    assert decide(svc.run(data), assess_quality(data), "").task is Task.UNCLEAR


def test_blurred_page_routes_to_vision(svc):
    data = _png(_printed_page().filter(ImageFilter.GaussianBlur(5)))
    d = decide(svc.run(data), assess_quality(data), "")
    assert d.use_vision


def test_photo_of_page_on_dark_desk_routes_to_tesseract(svc):
    im = Image.new("RGB", (1800, 1200), (90, 70, 50))
    im.paste(_printed_page(), (200, 250))
    data = _png(im)
    ocr = svc.run(data)
    d = decide(ocr, assess_quality(data), "")
    assert d.task is Task.TEXT_ONLY, (d.reason, ocr.text_coverage)


def test_labelled_diagram_routes_to_vision(svc):
    im = Image.new("RGB", (1200, 900), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((350, 250, 850, 750), outline="black", width=5)
    d.line((600, 250, 600, 750), fill="black", width=3)
    font = ImageFont.load_default(size=40)
    for text, xy in [("Nucleus", (60, 300)), ("Cell wall", (900, 300)), ("Parts of a plant cell", (300, 60))]:
        d.text(xy, text, fill="black", font=font)
    data = _png(im)
    assert decide(svc.run(data), assess_quality(data), "").use_vision


def test_table_routes_to_vision(svc):
    im = Image.new("RGB", (1200, 700), "white")
    d = ImageDraw.Draw(im)
    for r in range(5):
        d.line((50, 50 + r * 140, 1150, 50 + r * 140), fill="black", width=2)
    for c in range(4):
        d.line((50 + c * 366, 50, 50 + c * 366, 610), fill="black", width=2)
    for r in range(4):
        for c in range(3):
            d.text((70 + c * 366, 90 + r * 140), f"Row {r} value {c}", fill="black", font=ImageFont.load_default(size=34))
    data = _png(im)
    assert decide(svc.run(data), assess_quality(data), "").use_vision


def test_graph_with_readable_label_still_routes_to_vision(svc):
    data = _png(_graph())
    ocr = svc.run(data)
    assert "slope" in ocr.text.lower()
    assert decide(ocr, assess_quality(data), "find the slope").task is Task.GRAPH
    assert decide(ocr, assess_quality(data), "").use_vision  # few words / low coverage without a message

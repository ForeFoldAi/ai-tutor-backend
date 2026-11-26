"""Junk-figure detection: QR codes, QR banners, decorations recurring across chapters."""

from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from app.services.image_service.junk_figure_filter import (
    REASON_QR_BANNER,
    REASON_QR_CODE,
    BookFigure,
    classify_qr_junk,
    decode_qr_codes,
    find_recurring_decorations,
    matching_sibling_keys,
    phash_distance,
)
from app.services.image_service.textbook_image_extraction import compute_phash

cv2 = pytest.importorskip("cv2")


def _png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _qr_image(text: str = "http://epathshala.nic.in/QR/?id=1062CH01", size: int = 150) -> Image.Image:
    encoder = cv2.QRCodeEncoder.create()
    mat = encoder.encode(text)
    qr = Image.fromarray(mat).convert("RGB").resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (size + 20, size + 20), "white")
    canvas.paste(qr, (10, 10))
    return canvas


def _triangle_diagram(w: int = 300, h: int = 240) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.polygon([(40, 200), (260, 200), (150, 30)], outline=(0, 150, 220), width=3)
    for (x, y), t in (((30, 205), "B"), ((262, 205), "C"), ((146, 12), "A")):
        d.text((x, y), t, fill="black")
    return img


def test_standalone_qr_is_qr_code():
    data = _png(_qr_image())
    hits = decode_qr_codes(data)
    assert hits and hits[0].payload.startswith("http://epathshala")
    assert classify_qr_junk(data, figure_number=None) == REASON_QR_CODE


def test_banner_with_small_qr_is_qr_banner():
    banner = Image.new("RGB", (700, 320), (40, 170, 230))
    banner.paste(_qr_image(size=110), (60, 120))
    assert classify_qr_junk(_png(banner), figure_number=None) == REASON_QR_BANNER


def test_teaching_diagram_is_not_junk():
    assert decode_qr_codes(_png(_triangle_diagram())) == []
    assert classify_qr_junk(_png(_triangle_diagram()), figure_number=None) is None


def test_numbered_figure_is_never_qr_junk():
    assert classify_qr_junk(_png(_qr_image()), figure_number="6.14") is None


def test_corrupt_bytes_are_not_junk():
    assert classify_qr_junk(b"not an image", figure_number=None) is None


def test_compute_phash_returns_int_for_valid_image():
    # Regression: int(ImageHash) raised TypeError, so every phash was stored as NULL.
    ph = compute_phash(_png(_triangle_diagram()))
    assert isinstance(ph, int)
    assert -(1 << 63) <= ph < (1 << 63)


def test_phash_distance_handles_signed_values():
    assert phash_distance(-1, -1) == 0
    assert phash_distance(0, -1) == 64


def _fig(key, upload, ph, num=None):
    return BookFigure(key=key, upload_id=upload, phash=ph, figure_number=num)


def test_recurring_across_three_chapters_is_flagged():
    mascot = 0x0F0F_0F0F_0F0F_0F0F
    near = mascot ^ 0b111  # 3 bits away
    figs = [_fig(1, 10, mascot), _fig(2, 11, near), _fig(3, 12, mascot), _fig(4, 12, 0x7FFF_0000_0000_0000)]
    assert find_recurring_decorations(figs) == {1, 2, 3}


def test_recurring_in_only_two_chapters_is_kept():
    mascot = 0x0F0F_0F0F_0F0F_0F0F
    figs = [_fig(1, 10, mascot), _fig(2, 11, mascot), _fig(3, 11, mascot)]
    assert find_recurring_decorations(figs) == set()


def test_recurring_group_with_a_figure_number_is_kept():
    reused = 0x1234_5678_9ABC_DEF0
    figs = [_fig(1, 10, reused), _fig(2, 11, reused, "2.1"), _fig(3, 12, reused)]
    assert find_recurring_decorations(figs) == set()


def test_matching_sibling_keys_counts_distinct_uploads():
    ph = 0x00FF_00FF_00FF_00FF
    siblings = [_fig(7, 10, ph), _fig(8, 10, ph ^ 1), _fig(9, 11, ph), _fig(10, 12, ~ph)]
    uploads, keys = matching_sibling_keys(ph, siblings)
    assert uploads == {10, 11}
    assert sorted(keys) == [7, 8, 9]
    assert matching_sibling_keys(None, siblings) == (set(), [])

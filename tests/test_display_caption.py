"""Student-friendly display captions from PDF text layer (any subject)."""

from types import SimpleNamespace

from app.services.image_service.pdf_figure_context import resolve_display_caption


def _im(**kwargs):
    defaults = {
        "caption": "33",
        "figure_number": "2.6",
        "page_index": 6,
        "image_type": "diagram",
        "upload": SimpleNamespace(
            file_path="uploads/CBSE/CLASS_9/Social/c75c7c82_gees102.pdf"
        ),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_rain_gauge_caption_from_pdf():
    cap = resolve_display_caption(_im(), subtopic="Precipitation")
    assert "rain gauge" in cap.lower() or "2.6" in cap
    assert len(cap) >= 12


def test_wind_caption_from_pdf():
    cap = resolve_display_caption(
        _im(caption="wind sock garbage", figure_number="2.10", page_index=10),
        subtopic="Wind",
    )
    assert "wind" in cap.lower()
    assert len(cap) >= 15


def test_subtopic_fallback_without_hardcoded_labels():
    cap = resolve_display_caption(
        _im(caption="x", figure_number="2.8", page_index=8),
        subtopic="Atmospheric pressure",
    )
    assert "atmospheric pressure" in cap.lower() or "2.8" in cap

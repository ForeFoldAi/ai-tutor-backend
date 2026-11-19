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
    # Prefer PDF "rain gauge" when the textbook PDF is available; else subtopic.
    assert "rain gauge" in cap.lower() or "precipitation" in cap.lower() or "2.6" in cap
    assert len(cap) >= 12
    assert "be co" not in cap.lower()


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


def test_letter_spaced_ocr_caption_rejected():
    """OCR crumb like Fig 2.2's stored caption must never reach the student."""
    cap = resolve_display_caption(
        _im(
            caption="be co h l l h l l h Fig. 2.2",
            figure_number="2.2",
            page_index=1,
            generated_caption="",
            chapter_title="Chapter 2 — Understanding the Weather",
        ),
        subtopic="What is weather?",
    )
    assert "be co" not in cap.lower()
    assert "h l l" not in cap.lower()
    assert len(cap) >= 4
    # Prefer nearby PDF/subtopic over garbage
    assert "weather" in cap.lower() or "illustration" in cap.lower()


def test_prefers_generated_over_corrupt_ocr():
    cap = resolve_display_caption(
        _im(
            caption="Fig. 2.2—be co h l l h l l h Fig.",
            figure_number="2.2",
            page_index=1,
            generated_caption="Illustration showing weather as atmosphere state",
            upload=SimpleNamespace(file_path=""),  # force no PDF backfill
        ),
        subtopic=None,
    )
    assert "atmosphere" in cap.lower() or "weather" in cap.lower()
    assert "be co" not in cap.lower()

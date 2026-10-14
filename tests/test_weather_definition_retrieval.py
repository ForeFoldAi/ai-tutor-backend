"""Regression: 'what is weather' must return Fig 2.2, not OCR garbage figures."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.image_service.figure_context_gates import (
    is_core_definition_figure,
    is_corrupt_ml_caption,
)
from app.services.image_service.pdf_figure_context import page_supports_core_definition


def test_corrupt_caption_detects_garbage():
    assert is_corrupt_ml_caption("perating \nweather \nrection")
    assert is_corrupt_ml_caption("and and the People \nWeather\ngetting cold.")
    assert is_corrupt_ml_caption("be co h l l h l l h Fig. 2.2")
    assert is_corrupt_ml_caption("Fig. 2.2—be co h l l h l l h Fig.")
    assert not is_corrupt_ml_caption("Fig. 2.4.2. Cloudy weather")
    assert not is_corrupt_ml_caption("Rain gauge used to measure precipitation")


def test_page_supports_weather_definition():
    path = "/Applications/ForeFold/virtual tutor/ai-tutor-backend/uploads/CBSE/CLASS_9/Social/c75c7c82_gees102.pdf"
    assert page_supports_core_definition(path, 1, "weather")


def test_fig_22_core_definition_with_pdf_backfill():
    im = SimpleNamespace(
        figure_number="2.2",
        page_index=1,
        caption="several layers. The layer clos",
        nearby_text_before_figure="",
        nearby_text_after_figure="",
        upload=SimpleNamespace(
            file_path="/Applications/ForeFold/virtual tutor/ai-tutor-backend/uploads/CBSE/CLASS_9/Social/c75c7c82_gees102.pdf"
        ),
    )
    assert is_core_definition_figure(im, "weather")

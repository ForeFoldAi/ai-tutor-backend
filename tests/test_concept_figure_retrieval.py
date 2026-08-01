"""Concept→figure retrieval: synonym expansion + hard-filter for measurement visuals."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.image_service.image_intent_extractor import (
    extract_image_intent,
    _expand_concept_aliases,
)
from app.services.image_service.textbook_image_retrieval import (
    _concept_specificity_score,
    _hard_concept_filter,
)


def _fake_figure(**kwargs):
    defaults = dict(
        caption="",
        caption_normalized=None,
        generated_caption=None,
        figure_number=None,
        image_type="unknown",
        educational_role="primary_concept",
        section_title=None,
        subsection_title=None,
        nearby_text_before_figure="",
        nearby_text_after_figure="",
        figure_context="",
        semantic_keywords="",
        concept_tags="",
        educational_tags="",
        content_kind="figure",
        page_index=0,
        file_name="figures/fig.jpg",
        is_decorative=False,
        chapter_title=None,
        page_text_snippet=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_precipitation_intent_includes_rain_gauge_aliases():
    intent = extract_image_intent("what is precipitation?", [])
    terms = {t.lower() for t in intent.required_terms}
    assert "precipitation" in terms
    assert "rain gauge" in terms
    assert "rainfall" in terms
    # bare "rain" stays supporting, not the only distinctive required term
    assert "rain" not in terms or "rain gauge" in terms
    assert "instrument" in intent.preferred_types
    assert "diagram" in intent.preferred_types


def test_humidity_intent_includes_hygrometer():
    intent = extract_image_intent("what is humidity?", [])
    terms = {t.lower() for t in intent.required_terms}
    assert "humidity" in terms
    assert "hygrometer" in terms


def test_rain_gauge_figure_passes_precipitation_hard_filter():
    intent = extract_image_intent("what is precipitation?", [])
    im = _fake_figure(
        caption="Fig. 2.6. Rain gauge",
        caption_normalized="fig. 2.6. rain gauge",
        figure_number="2.6",
        image_type="instrument",
        nearby_text_before_figure="The amount of rainfall is measured with a rain gauge.",
    )
    reject, reason = _hard_concept_filter(intent, im)
    assert not reject, reason
    assert _concept_specificity_score(intent, im) >= 3.0


def test_hygrometer_figure_passes_humidity_hard_filter():
    intent = extract_image_intent("what is humidity?", [])
    im = _fake_figure(
        caption="Fig. 3.2. Hygrometer",
        caption_normalized="fig. 3.2. hygrometer",
        figure_number="3.2",
        image_type="instrument",
    )
    reject, reason = _hard_concept_filter(intent, im)
    assert not reject, reason
    assert _concept_specificity_score(intent, im) >= 3.0


def test_weather_definition_still_excludes_instruments():
    intent = extract_image_intent("what is weather?", [])
    assert "instrument" in intent.excluded_types
    assert "weather_station" in intent.excluded_types
    # Must not expand weather → rain gauge / AWS aliases
    terms = {t.lower() for t in intent.required_terms}
    assert "rain gauge" not in terms
    assert "aws" not in terms
    assert "weather station" not in terms

    rain_gauge = _fake_figure(
        caption="Fig. 2.6. Rain gauge",
        caption_normalized="fig. 2.6. rain gauge",
        figure_number="2.6",
        image_type="instrument",
    )
    reject, _reason = _hard_concept_filter(intent, rain_gauge)
    assert reject


def test_expand_aliases_reverse_lookup_precipitation():
    required, supporting = _expand_concept_aliases("precipitation")
    assert "rain gauge" in required
    assert "rainfall" in required
    assert "rain" in supporting

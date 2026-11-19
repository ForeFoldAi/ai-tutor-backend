"""Tests for ML topic tagging on tables and formulas."""

from __future__ import annotations

from app.services.image_service.structured_asset_tagger import (
    _merge_concept_tags,
    _regex_concept_hits,
    detect_section_from_page_text,
    tag_structured_asset_ml,
)


def test_detect_section_from_page_text():
    md = "# Chapter intro\n\n2.3 Weather Instruments\n\nSome rainfall data here."
    assert detect_section_from_page_text(md) == "Weather Instruments"


def test_regex_concept_hits_weather_table():
    blob = "rainfall temperature humidity monsoon wind speed comparison data table"
    hits = _regex_concept_hits(blob)
    assert "weather_climate" in hits
    assert "data_table" in hits


def test_merge_concept_tags_prefers_bge_then_regex():
    merged = _merge_concept_tags(
        [("weather_climate", 55.0), ("landforms", 30.0)],
        ["data_table"],
        min_sim=42.0,
    )
    assert merged[0] == "weather_climate"
    assert "data_table" in merged


def test_tag_structured_asset_ml_table_topics():
    result = tag_structured_asset_ml(
        content_kind="table",
        caption="Table 2.1 Monthly rainfall in selected cities",
        structured_content="City | Jan | Feb | Mar\nDelhi | 12 | 15 | 20",
        chapter_title="Climate of India",
        section_title="Rainfall Patterns",
        subject="Geography",
        grade_level="8",
        page_markdown="2.3 Rainfall Patterns\nCompare monsoon rainfall across regions.",
        figure_number="2.1",
    )
    assert result["concept_tags"]
    assert "table" in result["educational_tags"] or "data_table" in result["concept_tags"]
    assert result["title"]
    assert result["figure_context"]
    assert "rainfall" in result["semantic_keywords"].lower() or "weather" in result["semantic_keywords"].lower()


def test_tag_structured_asset_ml_formula_topics():
    result = tag_structured_asset_ml(
        content_kind="formula",
        structured_content="F = m * a",
        chapter_title="Forces and Motion",
        subject="Physics",
        page_markdown="3.1 Newton's Laws\nforce velocity acceleration",
    )
    assert result["concept_tags"]
    assert result["title"].lower().startswith("formula") or "force" in result["title"].lower()

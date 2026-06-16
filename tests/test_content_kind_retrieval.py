"""Tests for table/formula content-kind retrieval routing."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.image_service.content_kind_retrieval import (
    asset_retrieval_text,
    get_content_kind,
    referenced_asset_matches,
    resolve_content_kind_pool,
)
from app.services.image_service.image_intent_extractor import extract_image_intent


def _im(**kwargs):
    defaults = {
        "content_kind": "figure",
        "image_type": "diagram",
        "caption": "Fig 2.1 diagram",
        "file_name": "figures/fig_2_1.jpg",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_get_content_kind_table():
    assert get_content_kind(_im(content_kind="table", image_type="table")) == "table"


def test_resolve_pool_default_figures_only():
    images = [
        _im(content_kind="figure"),
        _im(content_kind="table", file_name="tables/t1.jpg"),
        _im(content_kind="formula", file_name="formulas/f1.jpg"),
    ]
    intent = extract_image_intent("what is weather", [])
    pool = resolve_content_kind_pool(images, intent)
    assert len(pool) == 1
    assert get_content_kind(pool[0]) == "figure"


def test_resolve_pool_table_query():
    images = [
        _im(content_kind="figure"),
        _im(content_kind="table", caption="Rainfall data", concept_tags="weather_climate|data_table"),
        _im(content_kind="formula"),
    ]
    intent = extract_image_intent("show the table with rainfall data", [])
    pool = resolve_content_kind_pool(images, intent)
    assert all(get_content_kind(im) == "table" for im in pool)
    assert len(pool) == 1


def test_resolve_pool_formula_query():
    intent = extract_image_intent("write the formula for force", [])
    assert "formula" in intent.preferred_content_kinds
    assert intent.query_type == "formula_request"


def test_referenced_table_match():
    intent = extract_image_intent("explain table 2.1", [])
    assert intent.referenced_asset_number == "2.1"
    assert intent.referenced_asset_kind == "table"
    im = _im(content_kind="table", figure_number="2.1", caption="Table 2.1 Cities")
    assert referenced_asset_matches(intent, im) is True


def test_asset_retrieval_text_includes_structured():
    im = _im(
        content_kind="table",
        structured_content="City|Rain|Temp",
        concept_tags="weather_climate",
        semantic_keywords="rainfall|monsoon",
    )
    blob = asset_retrieval_text(im)
    assert "city" in blob
    assert "weather" in blob

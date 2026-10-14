"""Section heading scope detection."""

from langchain_core.documents import Document

from app.services.section_heading import (
    enrich_chunks_with_section_metadata,
    extract_subtopics_from_scope,
    parse_heading_line,
    parse_section_hint,
    resolve_heading_scope,
    subtopics_for_main_section,
    title_match_score,
    HeadingScope,
    HeadingInfo,
)


def test_parse_numbered_heading():
    assert parse_section_hint("2.3 Weather Instruments") == ("2.3", "Weather Instruments", 1)


def test_pedagogy_signal_boxes_are_not_headings():
    assert parse_heading_line("No Warning") is None
    assert parse_heading_line("Warning (Take Action)") is None
    assert parse_heading_line("Watch (Be Updated)") is None
    assert parse_heading_line("2.4 Cyclones") == ("2.4", "Cyclones", 1)


def test_query_matches_main_section_with_children():
    chunks = [
        Document(
            page_content="Thermometer measures temperature.",
            metadata={"section_hint": "2.3.1 Thermometer", "page": 5},
        ),
        Document(
            page_content="Barometer measures pressure.",
            metadata={"section_hint": "2.3.2 Barometer", "page": 6},
        ),
        Document(
            page_content="Overview of instruments.",
            metadata={"section_hint": "2.3 Weather Instruments", "page": 4},
        ),
    ]
    enrich_chunks_with_section_metadata(chunks)
    scope = resolve_heading_scope("what is Weather Instruments", chunks)
    assert scope.kind == "main_section"
    assert scope.matched is not None
    assert scope.matched.title == "Weather Instruments"
    assert len(scope.child_headings or []) >= 2


def test_query_matches_subsection_only():
    chunks = [
        Document(
            page_content="Rain gauge collects precipitation.",
            metadata={"section_hint": "2.3.2 Rain Gauge", "page": 8},
        ),
        Document(
            page_content="Other instruments.",
            metadata={"section_hint": "2.3 Weather Instruments", "page": 4},
        ),
    ]
    enrich_chunks_with_section_metadata(chunks)
    scope = resolve_heading_scope("what is Rain Gauge", chunks)
    assert scope.kind == "subsection"
    assert scope.matched is not None
    assert "rain" in scope.matched.normalized_title


def test_title_match_score():
    assert title_match_score("what is weather instruments", "Weather Instruments") >= 85


def test_lettered_subtopics_use_textbook_titles():
    chunks = [
        Document(
            page_content="Weather Instruments\na) Temperature\nFig. 2.4 Thermometer",
            metadata={"page": 4},
        ),
        Document(
            page_content="b) Precipitation\nRain gauge measures rainfall.\nFig. 2.6 Rain gauge",
            metadata={"page": 7},
        ),
        Document(
            page_content="c) Atmospheric pressure\nFig. 2.7 Barometer",
            metadata={"page": 8},
        ),
        Document(
            page_content="d) Wind\nFig. 2.9 Anemometer",
            metadata={"page": 9},
        ),
        Document(
            page_content="e) Humidity\nFig. 2.10 Hygrometer",
            metadata={"page": 10},
        ),
        Document(
            page_content="Weather Stations",
            metadata={"section_hint": "Weather Stations", "page": 12},
        ),
    ]
    scope = HeadingScope(
        kind="main_section",
        matched=HeadingInfo(
            raw_hint="Weather Instruments",
            section_number="",
            title="Weather Instruments",
            level=1,
            page=4,
        ),
        page_start=4,
        page_end=12,
    )
    subs = extract_subtopics_from_scope(chunks, scope)
    titles = [s.title for s in subs]
    assert "Precipitation" in titles
    assert "Rain Gauge" not in titles
    assert subtopics_for_main_section(scope, chunks) == titles
    assert len(subs) == 5
    assert subs[1].title == "Precipitation"
    assert "2.6" in subs[1].figure_numbers


def test_precipitation_query_is_subsection_not_rain_gauge():
    chunks = [
        Document(
            page_content="b) Precipitation\nRain gauge.",
            metadata={"page": 7},
        ),
        Document(
            page_content="a) Temperature",
            metadata={"page": 4},
        ),
        Document(
            page_content="Overview.",
            metadata={"section_hint": "2.3 Weather Instruments", "page": 4},
        ),
    ]
    scope = resolve_heading_scope("what is Precipitation", chunks)
    assert scope.kind == "subsection"
    assert scope.matched is not None
    assert scope.matched.title == "Precipitation"

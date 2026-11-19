"""PDF text-layer fallback restores heading scope when Chroma chunks are sparse."""

from langchain_core.documents import Document

from app.services.pdf_text_layer import is_sparse_extracted_text
from app.services.section_heading import resolve_heading_scope, subtopics_for_main_section


def test_sparse_ml_markdown_detected():
    assert is_sparse_extracted_text("# \n    \n#")
    assert not is_sparse_extracted_text("Weather Instruments\na) Temperature\n" * 5)


def test_weather_instruments_heading_from_pdf_style_chunks():
    chunks = [
        Document(
            page_content=(
                "2 – Understanding the Weather\nWeather Instruments\n"
                "a) Temperature\nFig. 2.4.1 Snow melts\nFig. 2.5 Thermometer"
            ),
            metadata={"page": 4},
        ),
        Document(
            page_content="b) Precipitation\nRain gauge (Fig. 2.6).",
            metadata={"page": 6},
        ),
        Document(
            page_content="c) Atmospheric pressure\nFig. 2.7 Barometer",
            metadata={"page": 7},
        ),
        Document(
            page_content="d) Wind\nFig. 2.9 Anemometer",
            metadata={"page": 9},
        ),
        Document(
            page_content="e) Humidity\nHygrometer measures moisture.",
            metadata={"page": 10},
        ),
        Document(
            page_content="Weather Stations\nAutomated weather stations collect data.",
            metadata={"page": 12},
        ),
    ]
    scope = resolve_heading_scope("what are the Weather Instruments", chunks)
    assert scope.kind == "main_section"
    assert scope.matched is not None
    assert scope.matched.title == "Weather Instruments"
    subs = subtopics_for_main_section(scope, chunks)
    assert subs == [
        "Temperature",
        "Precipitation",
        "Atmospheric pressure",
        "Wind",
        "Humidity",
    ]

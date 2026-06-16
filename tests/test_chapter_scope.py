from unittest.mock import MagicMock, patch

from app.services.chapter_scope import (
    chapter_number_from_name,
    chapter_scope_mismatch_message,
    extract_mentioned_chapter_numbers,
    selected_chapter_numbers,
    substantive_query_terms,
    topic_chapter_mismatch_message,
)


def test_chapter_number_from_name():
    assert chapter_number_from_name("Chapter 2 - Understanding the Weather") == 2
    assert chapter_number_from_name("CHAPTER 10") == 10
    assert chapter_number_from_name("Weather") is None


def test_extract_mentioned_chapter_numbers():
    assert extract_mentioned_chapter_numbers("Explain chapter 1 weather") == {1}
    assert extract_mentioned_chapter_numbers("What is humidity?") == set()


def test_substantive_query_terms():
    assert substantive_query_terms("what is desert?") == {"desert"}
    assert substantive_query_terms("what is humidity?") == {"humidity"}


def test_no_mismatch_when_same_chapter():
    names = ["Chapter 2 - Understanding the Weather"]
    assert chapter_scope_mismatch_message("Explain chapter 2 instruments", names) is None


def test_mismatch_when_different_chapter():
    names = ["Chapter 2 - Understanding the Weather"]
    msg = chapter_scope_mismatch_message("What are weather instruments in chapter 1?", names)
    assert msg is not None
    assert "Chapter 1" in msg
    assert "Chapter 2" in msg


def test_no_mismatch_when_multiple_chapters_selected():
    names = ["Chapter 1 - Introduction", "Chapter 2 - Understanding the Weather"]
    assert selected_chapter_numbers(names) == {1, 2}
    assert chapter_scope_mismatch_message("Tell me about chapter 1", names) is None


def _doc(text: str, upload_id: str):
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"textbook_upload_id": upload_id}
    return doc


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.vector_service.retrieve_from_collection")
def test_topic_mismatch_desert_in_chapter_one(mock_retrieve, mock_labels):
    ch1_id = "11111111-1111-1111-1111-111111111111"
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_labels.return_value = {
        ch1_id: "Chapter 1 - India: Climate, Vegetation and Wildlife",
        ch2_id: "Chapter 2 - Understanding the Weather",
    }
    mock_retrieve.return_value = [
        _doc("Deserts are dry regions with very little rainfall.", ch1_id),
        _doc("The Thar desert lies in Rajasthan.", ch1_id),
        _doc("Humidity measures moisture in the air.", ch2_id),
    ]

    msg = topic_chapter_mismatch_message(
        "what is desert?",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    )
    assert msg is not None
    assert "desert" in msg.lower()
    assert "Chapter 1" in msg


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.vector_service.retrieve_from_collection")
def test_no_topic_mismatch_for_weather_question(mock_retrieve, mock_labels):
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_labels.return_value = {ch2_id: "Chapter 2 - Understanding the Weather"}
    mock_retrieve.return_value = []

    msg = topic_chapter_mismatch_message(
        "what is humidity?",
        docs=[
            _doc("Humidity is the amount of water vapour in the air.", ch2_id),
            _doc("Weather instruments measure humidity and temperature.", ch2_id),
        ],
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    )
    assert msg is None

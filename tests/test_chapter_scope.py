from unittest.mock import MagicMock, patch

from app.services.chapter_scope import (
    ChapterCoverageLevel,
    ChapterScopeChoice,
    assess_chapter_coverage,
    chapter_number_from_name,
    chapter_scope_mismatch_message,
    detect_chapter_scope_choice,
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
    assert "How would you like to continue?" in msg


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
    assert "How would you like to continue?" in msg


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


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.chapter_scope._best_other_chapter")
def test_not_covered_when_no_term_in_selected_chapter(mock_other, mock_labels):
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_labels.return_value = {ch2_id: "Chapter 2 - Understanding the Weather"}
    mock_other.return_value = ("", 0, 0)

    assessment = assess_chapter_coverage(
        "what is democracy?",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    )
    assert assessment.level == ChapterCoverageLevel.NONE


@patch("app.services.chapter_scope._best_other_chapter")
def test_full_coverage_when_term_in_chapter_title(mock_other):
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_other.return_value = ("", 0, 0)

    assessment = assess_chapter_coverage(
        "what is weather?",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    )
    assert assessment.level == ChapterCoverageLevel.FULL


@patch("app.services.chapter_scope._best_other_chapter")
def test_full_coverage_for_tell_about_this_chapter(mock_other):
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_other.return_value = ("", 0, 0)

    assessment = assess_chapter_coverage(
        "tell about this chapter",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    )
    assert assessment.level == ChapterCoverageLevel.FULL


def test_scope_choice_general():
    history = [
        {"role": "user", "content": "what is democracy?"},
        {
            "role": "assistant",
            "content": "How would you like to continue?\na) Stay\nb) Switch\nc) General",
        },
    ]
    assert detect_chapter_scope_choice("c", history) == ChapterScopeChoice.GENERAL
    assert detect_chapter_scope_choice("general explanation", history) == ChapterScopeChoice.GENERAL
    assert detect_chapter_scope_choice("I'll go with option B", history) == ChapterScopeChoice.SWITCH
    assert detect_chapter_scope_choice("I choose option c please", history) == ChapterScopeChoice.GENERAL


@patch("app.services.chapter_scope._best_other_chapter")
def test_pedagogical_quiz_skips_scope(mock_other):
    from app.services.chapter_scope import assess_chapter_coverage

    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_other.return_value = ("", 0, 0)
    assessment = assess_chapter_coverage(
        "Conduct quiz",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    )
    assert assessment.level == ChapterCoverageLevel.FULL


def test_related_follow_up_ozone_protection():
    from app.services.chapter_scope import is_related_chapter_follow_up

    history = [
        {
            "role": "assistant",
            "content": (
                "The ozone layer is a special part of the Earth’s atmosphere that protects "
                "life by absorbing most of the sun’s harmful ultraviolet rays."
            ),
        }
    ]
    assert is_related_chapter_follow_up("How is ozone layer protected?", history)

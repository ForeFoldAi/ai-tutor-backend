import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chapter_scope import (
    ChapterCoverageLevel,
    ChapterScopeChoice,
    assess_chapter_coverage,
    chapter_number_from_name,
    chapter_scope_mismatch_message,
    content_substantive_terms,
    detect_chapter_scope_choice,
    extract_mentioned_chapter_numbers,
    is_meta_only_follow_up,
    resolve_chapter_awareness_turn,
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


def test_additional_examples_request_is_meta_only():
    """Regression: 'give me additional examples' has no topic identity of its
    own — it must stay a meta follow-up (answered in the current chapter),
    not be scored as a new topic term that retrieval can misroute elsewhere."""
    assert content_substantive_terms("can you give me additional examples") == set()
    assert is_meta_only_follow_up("can you give me additional examples") is True
    assert is_meta_only_follow_up("give me some more examples") is True
    assert is_meta_only_follow_up("can you give another example") is True


def test_example_request_with_real_topic_keeps_scope_check():
    """A concrete topic mentioned alongside 'example' must still be treated
    as substantive, so genuine out-of-chapter questions are still caught."""
    assert content_substantive_terms("give me an example of photosynthesis") == {"photosynthesis"}
    assert is_meta_only_follow_up("give me an example of photosynthesis") is False


def test_i_dont_know_is_meta_only():
    """Regression: 'I don't know' / 'not sure' / 'idk' are uncertainty
    responses, not topic queries — they must not fall through to chapter
    coverage scoring on their leftover words ('know', 'sure')."""
    for phrase in ["i dont know", "i don't know", "not sure", "I'm not sure", "idk", "no idea"]:
        assert is_meta_only_follow_up(phrase) is True, phrase


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


@patch("app.services.chapter_scope._best_other_chapter")
def test_british_summarise_this_chapter_is_in_scope(mock_other):
    """'summarise' (UK) must not be treated as an out-of-chapter topic."""
    ch1_id = "11111111-1111-1111-1111-111111111111"
    mock_other.return_value = ("", 0, 0)

    for query in ("Summarise this chapter", "Summarize this chapter", "summary of this chapter"):
        assessment = assess_chapter_coverage(
            query,
            docs=[_doc("A square has four equal sides.", ch1_id)],
            collection_name="CBSE_CLASS_8_Mathematics",
            chapter_ids=[ch1_id],
            chapter_names=["Chapter 1 - A Square and A Cube"],
            board="CBSE",
            class_level="CLASS_8",
            subject_name="Mathematics",
        )
        assert assessment.level == ChapterCoverageLevel.FULL, query


@patch("app.services.chapter_scope._best_other_chapter")
def test_what_is_a_square_matches_chapter_title(mock_other):
    ch1_id = "11111111-1111-1111-1111-111111111111"
    mock_other.return_value = ("", 0, 0)

    assessment = assess_chapter_coverage(
        "What is a square?",
        docs=[],
        collection_name="CBSE_CLASS_8_Mathematics",
        chapter_ids=[ch1_id],
        chapter_names=["Chapter 1 - A Square and A Cube"],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Mathematics",
    )
    assert assessment.level == ChapterCoverageLevel.FULL
    assert substantive_query_terms("What is a square?") == {"square"}


@patch("app.services.chapter_scope._best_other_chapter")
def test_concept_related_math_problem_detects_square_practice(mock_other):
    from app.services.chapter_scope import (
        is_concept_related_math_problem,
        resolve_chapter_awareness_turn,
    )

    mock_other.return_value = ("", 0, 0)
    names = ["Chapter 1 - A Square and A Cube"]
    assert is_concept_related_math_problem(
        "A square field has side 25 m. Find its area.",
        subject_name="Mathematics",
        chapter_names=names,
    )
    assert is_concept_related_math_problem(
        "Find the cube of 12",
        subject_name="Mathematics",
        chapter_names=names,
    )
    # Unrelated algebra in a squares chapter → not concept-related
    assert not is_concept_related_math_problem(
        "Solve 2x + 3 = 11",
        subject_name="Mathematics",
        chapter_names=names,
    )
    # Non-math subject → never
    assert not is_concept_related_math_problem(
        "Find the area of a square of side 5",
        subject_name="Science",
        chapter_names=names,
    )

    early, _q, _a, guidance = asyncio.run(resolve_chapter_awareness_turn(
        "A square park has side 40 m. Find its perimeter.",
        docs=[],
        conversation_history=None,
        collection_name="CBSE_CLASS_8_Mathematics",
        chapter_ids=["11111111-1111-1111-1111-111111111111"],
        chapter_names=names,
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Mathematics",
    ))
    assert early is None
    assert "RELATED MATH PRACTICE" in guidance


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


def test_misheard_term_guess_finds_chapter_word():
    from app.services.chapter_scope import misheard_term_guess

    ch1_id = "11111111-1111-1111-1111-111111111111"
    docs = [_doc("Babur founded the Mughal empire after defeating the Delhi Sultanate.", ch1_id)]
    assert misheard_term_guess("tell me about the muggles", docs=docs, chapter_names=[]) == "mughal"
    # No plausible near-miss → no guess.
    assert misheard_term_guess("what is photosynthesis", docs=docs, chapter_names=[]) is None


@patch("app.services.chapter_scope._best_other_chapter")
def test_misheard_term_asks_confirmation_instead_of_wall(mock_other):
    """Rule 2: an out-of-chapter term that's a plausible ASR mishearing of a
    real chapter term must trigger a confirming question, not the a/b/c wall."""
    mock_other.return_value = ("", 0, 0)
    ch1_id = "11111111-1111-1111-1111-111111111111"
    names = ["Chapter 5 - Reshaping India's Political Map"]
    docs = [_doc("Babur founded the Mughal empire after defeating the Delhi Sultanate.", ch1_id)]

    early, _q, _a, _g = asyncio.run(resolve_chapter_awareness_turn(
        "tell me about the muggles",
        docs=docs,
        conversation_history=None,
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[ch1_id],
        chapter_names=names,
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    ))
    assert early is not None
    assert "did you mean" in early.lower()
    assert "Mughal" in early


@patch("app.services.chapter_scope._best_other_chapter")
def test_misheard_confirmation_yes_answers_the_real_term(mock_other):
    mock_other.return_value = ("", 0, 0)
    ch1_id = "11111111-1111-1111-1111-111111111111"
    names = ["Chapter 5 - Reshaping India's Political Map"]
    history = [
        {"role": "user", "content": "tell me about the muggles"},
        {"role": "assistant", "content": "I think I may have misheard — did you mean **Mughal**?"},
    ]

    early, effective_query, _a, guidance = asyncio.run(resolve_chapter_awareness_turn(
        "yes",
        docs=[],
        conversation_history=history,
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[ch1_id],
        chapter_names=names,
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    ))
    assert early is None
    assert effective_query == "Mughal"
    assert "GENERAL EXPLANATION" in guidance


@patch("app.services.chapter_scope._llm_confirms_topic_mismatch", new_callable=AsyncMock)
@patch("app.services.chapter_scope._best_other_chapter")
def test_misheard_confirmation_no_falls_back_to_literal_term(mock_other, mock_llm):
    mock_other.return_value = ("", 0, 0)
    mock_llm.return_value = True  # genuine mismatch once treated literally
    ch1_id = "11111111-1111-1111-1111-111111111111"
    names = ["Chapter 5 - Reshaping India's Political Map"]
    history = [
        {"role": "user", "content": "tell me about the muggles"},
        {"role": "assistant", "content": "I think I may have misheard — did you mean **Mughal**?"},
    ]

    early, effective_query, _a, _g = asyncio.run(resolve_chapter_awareness_turn(
        "no",
        docs=[],
        conversation_history=history,
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[ch1_id],
        chapter_names=names,
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    ))
    # Falls through to normal scope handling using the literal original term —
    # the a/b/c wall names the term the student actually said, not "no".
    assert early is not None
    assert "muggles" in early.lower()


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


def test_with_an_image_is_visual_follow_up_not_topic():
    from app.services.chapter_scope import (
        is_image_follow_up_request,
        resolve_chapter_awareness_turn,
    )
    from app.services.conversation_intent_classifier import (
        FollowupType,
        classify_followup_regex,
    )

    assert is_image_follow_up_request("with an image")
    assert classify_followup_regex("with an image") == FollowupType.ASK_VISUAL

    history = [
        {"role": "user", "content": "can you explain me about weather"},
        {
            "role": "assistant",
            "content": "Weather is how the air around us feels right now — hot, cold, windy, or rainy.",
        },
    ]
    early, effective, _assessment, guidance = asyncio.run(resolve_chapter_awareness_turn(
        "with an image",
        docs=[],
        conversation_history=history,
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=["22222222-2222-2222-2222-222222222222"],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    ))
    assert early is None
    assert "weather" in effective.lower()
    assert "VISUAL REQUEST" in guidance
    assert "a/b/c" in guidance.lower()


def test_additional_examples_never_reaches_coverage_scoring():
    """Regression at the real gate: assess_chapter_coverage short-circuits to
    FULL when `terms` is empty (chapter_scope.py:860) — this is what actually
    stops 'give me additional examples' from being scored against retrieval,
    not is_meta_only_follow_up (which nothing in the pipeline calls)."""
    assessment = assess_chapter_coverage(
        "can you give me additional examples",
        docs=[],
        collection_name="CBSE_CLASS_8_English",
        chapter_ids=["11111111-1111-1111-1111-111111111111"],
        chapter_names=["Unit 2 - Values and Dispositions"],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="English",
    )
    assert assessment.level == ChapterCoverageLevel.FULL


@patch("app.services.chapter_scope._llm_confirms_topic_mismatch", new_callable=AsyncMock)
@patch("app.services.chapter_scope._best_other_chapter")
def test_llm_second_opinion_suppresses_false_positive_wall(mock_other, mock_llm):
    """The permanent fix for the class of bug regex/keyword lists keep
    missing: when every heuristic above fails to recognize a follow-up, the
    LLM second opinion gets one chance to veto the disruptive wall."""
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_other.return_value = ("", 0, 0)
    mock_llm.return_value = False  # "this reads like a follow-up"

    early, effective, assessment, guidance = asyncio.run(resolve_chapter_awareness_turn(
        "what is democracy?",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        conversation_history=None,
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    ))
    assert early is None
    assert assessment.level == ChapterCoverageLevel.NONE
    mock_llm.assert_awaited_once()


@patch("app.services.chapter_scope._llm_confirms_topic_mismatch", new_callable=AsyncMock)
@patch("app.services.chapter_scope._best_other_chapter")
def test_llm_second_opinion_confirms_real_mismatch(mock_other, mock_llm):
    """A genuine topic mismatch still shows the wall — the LLM check only
    ever suppresses false positives, never invents new ones."""
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_other.return_value = ("", 0, 0)
    mock_llm.return_value = True  # "this is genuinely a new topic"

    early, *_rest = asyncio.run(resolve_chapter_awareness_turn(
        "what is democracy?",
        docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
        conversation_history=None,
        collection_name="CBSE_CLASS_9_Social",
        chapter_ids=[ch2_id],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
    ))
    assert early is not None
    assert "How would you like to continue?" in early


@patch("app.services.chapter_scope._best_other_chapter")
def test_llm_second_opinion_fails_safe_on_error(mock_other):
    """If the LLM call errors/times out, keep today's behavior (show the
    wall) rather than silently letting every retrieval miss through."""
    ch2_id = "22222222-2222-2222-2222-222222222222"
    mock_other.return_value = ("", 0, 0)

    with patch("app.services.llm_client.complete", side_effect=RuntimeError("boom")):
        early, *_rest = asyncio.run(resolve_chapter_awareness_turn(
            "what is democracy?",
            docs=[_doc("Humidity measures moisture in the air.", ch2_id)],
            conversation_history=None,
            collection_name="CBSE_CLASS_9_Social",
            chapter_ids=[ch2_id],
            chapter_names=["Chapter 2 - Understanding the Weather"],
            board="CBSE",
            class_level="CLASS_9",
            subject_name="Social",
        ))
    assert early is not None

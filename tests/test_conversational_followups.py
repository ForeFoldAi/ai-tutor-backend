"""Multi-subject conversational follow-ups must not trigger false Stay/Switch walls."""

import asyncio
from unittest.mock import MagicMock, patch

from app.services.chapter_scope import (
    ChapterCoverageLevel,
    assess_chapter_coverage,
    chapter_concept_terms,
    content_substantive_terms,
    is_concept_related_math_problem,
    is_meta_only_follow_up,
    resolve_chapter_awareness_turn,
    resolve_chapter_scope_with_retrieval,
    substantive_query_terms,
)
from app.services.conversation_context import FollowupType, resolve_conversation_context
from app.services.conversation_intent_classifier import classify_followup_regex

CH2 = "Chapter 2 - Reshaping India's Political Map"
CH2_ID = "22222222-2222-2222-2222-222222222222"
CH1 = "Chapter 1 - Natural Resources and Their Use"
CH1_ID = "11111111-1111-1111-1111-111111111111"

MATH_HISTORY = [
    {"role": "user", "content": "How do we simplify 2/4?"},
    {
        "role": "assistant",
        "content": "2/4 can be simplified to 1/2 because both numbers can be divided by 2.",
    },
]

SCIENCE_HISTORY = [
    {"role": "user", "content": "What is photosynthesis?"},
    {
        "role": "assistant",
        "content": "Plants need sunlight for photosynthesis to make food.",
    },
]

SOCIAL_HISTORY = [
    {"role": "user", "content": "Tell me about Akbar"},
    {
        "role": "assistant",
        "content": "Akbar expanded Mughal control over many regions in north India.",
    },
]

ENGLISH_HISTORY = [
    {"role": "user", "content": "What is an adjective?"},
    {"role": "assistant", "content": "Beautiful is an adjective — it describes a noun."},
]


def _doc(text: str, upload_id: str):
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"textbook_upload_id": upload_id}
    return doc


def test_meta_words_not_content_topics():
    assert content_substantive_terms("deep telling understand") == set()
    assert content_substantive_terms("explain this deeply") == set()
    assert content_substantive_terms("what is desert?") == {"desert"}


def test_meta_filter_does_not_break_curriculum_vocabulary():
    """Chapter titles / math concepts keep words like 'simple' and 'understanding'."""
    assert chapter_concept_terms(["Chapter 4 - Simple Equations"]) >= {"simple", "equations"}
    assert chapter_concept_terms(["Chapter 3 - Understanding Quadrilaterals"]) >= {
        "understanding",
        "quadrilaterals",
    }
    assert substantive_query_terms("what is mean and median") == {"mean", "median"}


def test_math_practice_still_detected():
    assert is_concept_related_math_problem(
        "Solve the simple equation 2x + 3 = 7",
        subject_name="Mathematics",
        chapter_names=["Chapter 4 - Simple Equations"],
    )


def test_deepen_phrases_classified_as_continue():
    assert classify_followup_regex("Explain this deeply.") == FollowupType.CONTINUE_EXPLANATION
    assert classify_followup_regex("Can you explain what you are telling me in deep?") == (
        FollowupType.CONTINUE_EXPLANATION
    )
    assert classify_followup_regex("How did you get this?") == FollowupType.CONTINUE_EXPLANATION


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.chapter_scope._best_other_chapter")
def test_deep_telling_understand_no_wall_without_history(mock_other, mock_labels):
    """Regression: STT junk must not invent Natural Resources chapter."""
    mock_other.return_value = (CH1_ID, 3, 1)
    mock_labels.return_value = {CH1_ID: CH1, CH2_ID: CH2}
    q = "deep telling understand"
    assert is_meta_only_follow_up(q)
    assessment = assess_chapter_coverage(
        q,
        docs=[_doc("Mughal expansion changed India's political map.", CH2_ID)],
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
    )
    assert assessment.level == ChapterCoverageLevel.FULL
    early, _, _, _ = asyncio.run(resolve_chapter_awareness_turn(
        q,
        docs=[_doc("Mughal expansion changed India's political map.", CH2_ID)],
        conversation_history=None,
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
    ))
    assert early is None
    assert "How would you like to continue" not in (early or "")


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.chapter_scope._best_other_chapter")
def test_explain_this_deeply_social_no_wall(mock_other, mock_labels):
    mock_other.return_value = (CH1_ID, 3, 1)
    mock_labels.return_value = {CH1_ID: CH1, CH2_ID: CH2}
    q = "Explain this deeply."
    ctx = resolve_conversation_context(q, conversation_history=SOCIAL_HISTORY, chapter=CH2)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert "akbar" in ctx.retrieval_query.lower() or "mughal" in ctx.retrieval_query.lower()
    early, _, _, guidance = asyncio.run(resolve_chapter_awareness_turn(
        q,
        docs=[_doc("Akbar expanded the Mughal empire.", CH2_ID)],
        conversation_history=SOCIAL_HISTORY,
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
        scope_query=ctx.retrieval_query,
    ))
    assert early is None
    assert "not covered" not in (guidance or "").lower()


def test_why_math_follow_up():
    ctx = resolve_conversation_context("Why?", conversation_history=MATH_HISTORY)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert "2/4" in ctx.retrieval_query or "simplif" in ctx.retrieval_query.lower()


def test_how_science_follow_up():
    ctx = resolve_conversation_context("How?", conversation_history=SCIENCE_HISTORY)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert "photosynthesis" in ctx.retrieval_query.lower() or "sunlight" in ctx.retrieval_query.lower()


def test_why_english_follow_up():
    ctx = resolve_conversation_context("Why?", conversation_history=ENGLISH_HISTORY)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert "adjective" in ctx.retrieval_query.lower() or "beautiful" in ctx.retrieval_query.lower()


def test_how_did_you_get_this_math():
    history = [
        {"role": "user", "content": "What is 6 times 4?"},
        {"role": "assistant", "content": "The answer is 24 because we multiply 6 by 4."},
    ]
    ctx = resolve_conversation_context("How did you get this?", conversation_history=history)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert "24" in ctx.retrieval_query or "multiply" in ctx.retrieval_query.lower()


def test_i_dont_understand_clarification():
    ctx = resolve_conversation_context("I don't understand.", conversation_history=SCIENCE_HISTORY)
    assert ctx.followup_type == FollowupType.CLARIFICATION.value


def test_give_me_an_example():
    ctx = resolve_conversation_context("Give me an example.", conversation_history=MATH_HISTORY)
    assert ctx.followup_type == FollowupType.ASK_EXAMPLE.value


@patch("app.services.section_retrieval.retrieve_for_tutor_query")
def test_voice_gate_skips_deepen_follow_up(mock_retrieve):
    mock_retrieve.return_value = ([], None, "")
    q = "Can you explain what you are telling me in deep?"
    conv = resolve_conversation_context(q, conversation_history=SOCIAL_HISTORY, chapter=CH2)
    msg = resolve_chapter_scope_with_retrieval(
        q,
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
        retrieval_query=conv.retrieval_query,
        conversation_history=SOCIAL_HISTORY,
    )
    assert msg is None


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.vector_service.retrieve_from_collection")
def test_true_ooc_still_walls(mock_retrieve, mock_labels):
    """True out-of-chapter content must still offer Stay/Switch."""
    mock_labels.return_value = {CH1_ID: CH1, CH2_ID: CH2}
    mock_retrieve.return_value = [
        _doc("Deserts are dry regions.", CH1_ID),
        _doc("Weather and humidity.", CH2_ID),
    ]
    from app.services.chapter_scope import topic_chapter_mismatch_message

    msg = topic_chapter_mismatch_message(
        "what is desert?",
        docs=[_doc("Humidity measures moisture.", CH2_ID)],
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[CH2_ID],
        chapter_names=["Chapter 2 - Understanding the Weather"],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    )
    assert msg is not None
    assert "How would you like to continue?" in msg


def test_how_many_maps_is_current_lesson_no_wall():
    """
    Regression: student clarifying about the political map/figures
    must not trigger Stay/Switch chapter walls.
    """
    from app.services.chapter_scope import resolve_chapter_awareness_turn

    q = "how many maps do we have"
    # Provide docs that would normally be used for coverage checks.
    # The key behavior is that chapter_scope should skip topic checks
    # for this query and therefore never emit the a/b/c wall menu.
    docs = [
        _doc("Fig 2.3 shows a political map snapshot for a period.", CH2_ID),
        _doc("Fig 2.12 and Fig 2.16 are other political map snapshots.", CH2_ID),
    ]
    early, _, _, guidance = asyncio.run(resolve_chapter_awareness_turn(
        q,
        docs=docs,
        conversation_history=None,
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
        scope_query=q,
    ))
    assert early is None
    assert guidance is None or "How would you like to continue?" not in guidance

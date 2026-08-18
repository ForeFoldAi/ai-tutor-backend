"""Follow-up conversation understanding — scope, context, and chapter continuity."""

from unittest.mock import MagicMock, patch

from app.services.chapter_scope import (
    ChapterCoverageLevel,
    assess_chapter_coverage,
    is_current_lesson_query,
    is_explicit_chapter_switch_request,
    is_session_continuation_follow_up,
    resolve_chapter_awareness_turn,
)
from app.services.conversation_context import (
    FollowupType,
    resolve_conversation_context,
    resolve_student_query_context,
)
from app.services.conversation_intent_classifier import classify_followup_regex


def _doc(text: str, upload_id: str):
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"textbook_upload_id": upload_id}
    return doc


MARATHAS_HISTORY = [
    {
        "role": "user",
        "content": "Explain Shivaji's successors and the Mughal attacks",
    },
    {
        "role": "assistant",
        "content": (
            "After Shivaji, Sambhaji and Rajaram led the Marathas. "
            "The Mughals launched several campaigns against them."
        ),
    },
]


CH3 = "Chapter 3 - The Rise of the Marathas"
CH3_ID = "33333333-3333-3333-3333-333333333333"
CH2 = "Chapter 2 - Reshaping India's Political Map"
CH2_ID = "22222222-2222-2222-2222-222222222222"


def test_tell_me_about_the_lesson_is_current_lesson_not_topic_search():
    q = "Can you tell me about the lesson?"
    assert is_current_lesson_query(q)
    ctx = resolve_conversation_context(q, chapter=CH2)
    assert "reshaping" in ctx.retrieval_query.lower() or "political" in ctx.retrieval_query.lower()
    assert "lesson" not in ctx.retrieval_query.lower().split()[-3:]


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.chapter_scope._best_other_chapter")
def test_tell_me_about_lesson_no_not_covered_wall(mock_other, mock_labels):
    """Regression: must not produce 'lesson is not covered in Chapter 2'."""
    mock_other.return_value = ("", 0, 0)
    mock_labels.return_value = {CH2_ID: CH2}
    q = "Can you tell me about the lesson?"
    early, effective, assessment, guidance = resolve_chapter_awareness_turn(
        q,
        docs=[_doc("India's political map changed after independence.", CH2_ID)],
        conversation_history=None,
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
        scope_query=resolve_conversation_context(q, chapter=CH2).retrieval_query,
    )
    assert early is None
    assert "not covered" not in (early or "").lower()
    assert "not covered" not in (guidance or "").lower()
    assert "How would you like to continue" not in (early or "")
    assert "reshaping" in effective.lower() or "political" in effective.lower() or "main concepts" in effective.lower()


def test_teach_this_lesson_is_current_lesson_without_history():
    assert is_current_lesson_query("Can you teach me this lesson?")
    assert is_session_continuation_follow_up("Can you teach me this lesson?", None)


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.chapter_scope._best_other_chapter")
def test_teach_this_lesson_no_not_covered_wall_without_history(mock_other, mock_labels):
    mock_other.return_value = ("", 0, 0)
    mock_labels.return_value = {CH2_ID: CH2}
    q = "Can you teach me this lesson?"
    conv = resolve_conversation_context(q, chapter=CH2)
    early, effective, _, guidance = resolve_chapter_awareness_turn(
        q,
        docs=[_doc("Political boundaries of India changed over time.", CH2_ID)],
        conversation_history=None,
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
        scope_query=conv.retrieval_query,
    )
    assert early is None
    assert "not covered" not in (guidance or "").lower()
    assert conv.intent_method == "current_lesson"
    assert effective


@patch("app.services.section_retrieval.retrieve_for_tutor_query")
def test_voice_scope_gate_lesson_query_returns_none(mock_retrieve):
    """Voice HTTP pre-check must not block current-lesson phrasing."""
    mock_retrieve.return_value = ([], None, "")
    q = "Can you tell me about the lesson?"
    conv = resolve_conversation_context(q, chapter=CH2)
    from app.services.chapter_scope import resolve_chapter_scope_with_retrieval

    msg = resolve_chapter_scope_with_retrieval(
        q,
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
        retrieval_query=conv.retrieval_query,
        conversation_history=None,
    )
    assert msg is None
    mock_retrieve.assert_not_called()


def test_assess_coverage_skips_lesson_meta_term():
    assessment = assess_chapter_coverage(
        "Can you tell me about the lesson?",
        docs=[],
        collection_name="CBSE_CLASS_7_Social",
        chapter_ids=[CH2_ID],
        chapter_names=[CH2],
        board="CBSE",
        class_level="CLASS_7",
        subject_name="Social",
    )
    assert assessment.level == ChapterCoverageLevel.FULL


def test_explain_this_with_history_is_follow_up():
    ctx = resolve_conversation_context("Explain this.", conversation_history=MARATHAS_HISTORY, chapter=CH3)
    assert ctx.followup_type in (
        FollowupType.CONTINUE_EXPLANATION.value,
        FollowupType.CLARIFICATION.value,
    )
    assert is_session_continuation_follow_up("Explain this.", MARATHAS_HISTORY)


def test_chapter_switch_explicit_only():
    assert is_explicit_chapter_switch_request("Teach me Chapter 4.")
    assert not is_explicit_chapter_switch_request("Can you tell me about the lesson?")


def test_summarize_key_points_is_follow_up_not_new_topic():
    ctx = resolve_conversation_context(
        "Can you summarize the key points?",
        conversation_history=MARATHAS_HISTORY,
        chapter=CH3,
    )
    assert ctx.followup_type == FollowupType.ASK_SUMMARY.value
    assert "shivaji" in ctx.retrieval_query.lower() or "maratha" in ctx.retrieval_query.lower()


@patch("app.services.chapter_scope._subject_upload_labels")
def test_summarize_key_points_no_not_covered_wall(mock_labels):
    mock_labels.return_value = {CH3_ID: CH3}
    early, effective, _assessment, guidance = resolve_chapter_awareness_turn(
        "Can you summarize the key points?",
        docs=[_doc("Shivaji established the Maratha kingdom.", CH3_ID)],
        conversation_history=MARATHAS_HISTORY,
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH3_ID],
        chapter_names=[CH3],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
        scope_query="Explain Shivaji's successors summary key points",
    )
    assert early is None
    assert "not covered" not in (guidance or "").lower()
    assert effective  # answers in session, not blocked


def test_why_is_follow_up():
    ctx = resolve_conversation_context("Why?", conversation_history=MARATHAS_HISTORY)
    assert ctx.followup_type == FollowupType.CONTINUE_EXPLANATION.value
    assert is_session_continuation_follow_up("Why?", MARATHAS_HISTORY)


def test_explain_again_simplification():
    ctx = resolve_conversation_context(
        "Can you explain that again in simple words?",
        conversation_history=MARATHAS_HISTORY,
    )
    assert ctx.followup_type in (
        FollowupType.SIMPLIFY.value,
        FollowupType.CLARIFICATION.value,
        FollowupType.CONTINUE_EXPLANATION.value,
    )
    assert is_session_continuation_follow_up(
        "Can you explain that again in simple words?",
        MARATHAS_HISTORY,
    )


def test_another_example_is_follow_up():
    assert classify_followup_regex("Give me another example") == FollowupType.ASK_EXAMPLE
    assert is_session_continuation_follow_up("Give me another example.", MARATHAS_HISTORY)


def test_explicit_chapter_switch_only_when_clear():
    assert is_explicit_chapter_switch_request("Teach me Chapter 4.")
    assert is_explicit_chapter_switch_request("Let's learn chapter 5")
    assert not is_explicit_chapter_switch_request("Can you summarize the key points?")
    assert not is_explicit_chapter_switch_request("Why did he do that?")


def test_teach_this_lesson_is_session_continuation():
    assert is_session_continuation_follow_up(
        "Can you teach me this lesson?",
        MARATHAS_HISTORY,
    )


def test_resolve_student_query_context_adapter():
    out = resolve_student_query_context(
        "Why?",
        conversation_history=MARATHAS_HISTORY,
        current_chapter=CH3,
        current_subject="Social",
        current_grade="CLASS_8",
    )
    assert out["intent"] == FollowupType.CONTINUE_EXPLANATION.value
    assert out["requires_rag"] is True
    assert "shivaji" in out["resolved_query"].lower() or "maratha" in out["resolved_query"].lower()


@patch("app.services.chapter_scope._subject_upload_labels")
@patch("app.services.chapter_scope._best_other_chapter")
def test_long_follow_up_still_continues_session(mock_other, mock_labels):
    mock_other.return_value = ("", 0, 0)
    mock_labels.return_value = {CH3_ID: CH3}
    long_q = (
        "I understood the first part about Shivaji, but I am confused about why "
        "the Mughal strategy worked and what happened to Sambhaji afterward. "
        "Can you explain that step by step with a simple example?"
    )
    assert is_session_continuation_follow_up(long_q, MARATHAS_HISTORY)
    early, _, assessment, guidance = resolve_chapter_awareness_turn(
        long_q,
        docs=[_doc("Sambhaji was captured by the Mughals.", CH3_ID)],
        conversation_history=MARATHAS_HISTORY,
        collection_name="CBSE_CLASS_8_Social",
        chapter_ids=[CH3_ID],
        chapter_names=[CH3],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
        scope_query=MARATHAS_HISTORY[0]["content"],
    )
    assert early is None
    assert guidance is not None
    assert assessment is None or assessment.level != ChapterCoverageLevel.NONE or guidance

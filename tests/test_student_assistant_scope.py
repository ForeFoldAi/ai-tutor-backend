"""Practice/ask/explain scope: chapter sticks across follow-ups; all-subject matching."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.main import app  # noqa: F401 — app import order avoids circular imports
from app.modules.student_assistant.service import (
    _norm,
    _title_match_score,
    detect_learning_scope,
)


def _overview():
    values_ch = SimpleNamespace(
        id=2,
        chapter="Unit 2 - Values and Dispositions",
        file_name="unit2.pdf",
    )
    grammar_ch = SimpleNamespace(
        id=3,
        chapter="English Grammar - Prepositions",
        file_name="prep.pdf",
    )
    resources_ch = SimpleNamespace(
        id=32,
        chapter="Chapter 1 - Natural Resources and Their Use",
        file_name="hees101.pdf",
    )
    political_ch = SimpleNamespace(
        id=33,
        chapter="Chapter 2 - Reshaping India's Political Map",
        file_name="hees102.pdf",
    )
    social = SimpleNamespace(
        subject_name="Social",
        board=SimpleNamespace(value="CBSE"),
        class_level=SimpleNamespace(value="Class_8"),
        chapters=[values_ch, grammar_ch, resources_ch, political_ch],
    )
    fractions_ch = SimpleNamespace(
        id=40,
        chapter="Chapter 2 - Fractions and Decimals",
        file_name="maths.pdf",
    )
    maths = SimpleNamespace(
        subject_name="Mathematics",
        board=SimpleNamespace(value="CBSE"),
        class_level=SimpleNamespace(value="Class_8"),
        chapters=[fractions_ch],
    )
    photo_ch = SimpleNamespace(
        id=50,
        chapter="Chapter 5 - Photosynthesis in Plants",
        file_name="science.pdf",
    )
    science = SimpleNamespace(
        subject_name="Science",
        board=SimpleNamespace(value="CBSE"),
        class_level=SimpleNamespace(value="Class_8"),
        chapters=[photo_ch],
    )
    return SimpleNamespace(subjects=[social, maths, science])


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_practice_followup_keeps_values_chapter_not_prepositions(mock_overview):
    mock_overview.return_value = _overview()
    history = [
        {
            "role": "user",
            "content": "Give me a practice problem on Unit 2 - Values and Dispositions",
        }
    ]
    scope = detect_learning_scope(
        MagicMock(),
        MagicMock(),
        "Give me practice questions on values and preposition",
        history,
        agent_mode="practice",
    )
    assert scope["chapter_name"] == "Unit 2 - Values and Dispositions"


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_practice_can_switch_when_new_chapter_named_clearly(mock_overview):
    mock_overview.return_value = _overview()
    history = [
        {
            "role": "user",
            "content": "Give me a practice problem on Unit 2 - Values and Dispositions",
        }
    ]
    scope = detect_learning_scope(
        MagicMock(),
        MagicMock(),
        "Give me practice on English Grammar - Prepositions",
        history,
        agent_mode="practice",
    )
    assert scope["chapter_name"] == "English Grammar - Prepositions"


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_ask_chapter_number_switches_from_chapter_one(mock_overview):
    """'social chapter 2' is an in-book ask — do not stay stuck on chapter 1."""
    mock_overview.return_value = _overview()
    history = [
        {
            "role": "user",
            "content": "What is important to know about Chapter 1 - Natural Resources and Their Use?",
        }
    ]
    scope = detect_learning_scope(
        MagicMock(),
        MagicMock(),
        "explain me social chapter 2",
        history,
        agent_mode="ask",
    )
    assert scope.get("chapter_name") == "Chapter 2 - Reshaping India's Political Map"


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_ask_cold_start_resolves_social_chapter_two(mock_overview):
    mock_overview.return_value = _overview()
    scope = detect_learning_scope(
        MagicMock(),
        MagicMock(),
        "explain me social chapter 2",
        agent_mode="ask",
    )
    assert scope.get("chapter_name") == "Chapter 2 - Reshaping India's Political Map"


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_ask_matches_nature_resource_heading_to_natural_resources_chapter(mock_overview):
    mock_overview.return_value = _overview()
    scope = detect_learning_scope(
        MagicMock(),
        MagicMock(),
        "when does Nature becomes A resource?",
        agent_mode="ask",
    )
    assert scope.get("chapter_name") == "Chapter 1 - Natural Resources and Their Use"


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_stem_match_works_for_math_and_science_across_modes(mock_overview):
    mock_overview.return_value = _overview()
    for mode in ("ask", "practice", "explain"):
        math_scope = detect_learning_scope(
            MagicMock(),
            MagicMock(),
            "explain this fraction problem",
            agent_mode=mode,
        )
        assert math_scope.get("chapter_name") == "Chapter 2 - Fractions and Decimals", mode

        sci_scope = detect_learning_scope(
            MagicMock(),
            MagicMock(),
            "how do plant photosynthesize?",
            agent_mode=mode,
        )
        assert sci_scope.get("chapter_name") == "Chapter 5 - Photosynthesis in Plants", mode


@patch("app.modules.student_assistant.service.learning_service.get_overview")
def test_content_fallback_resolves_chapter_when_title_tokens_miss(mock_overview):
    """Body-only wording (no chapter-title stem hit) still finds the chapter via RAG."""
    mock_overview.return_value = _overview()
    doc = SimpleNamespace(
        page_content=(
            "Leaves make food using sunlight through chlorophyll. "
            "This process feeds the plant."
        ),
        metadata={
            "textbook_upload_id": "50",
            "content_label": "Chapter 5 - Photosynthesis in Plants",
        },
    )

    def _retrieve(query, collection_name="", chapter_ids=None, k=6, **_kw):
        if collection_name.endswith("_Science") or "_Science" in collection_name:
            return [doc]
        return []

    with patch(
        "app.services.vector_service.retrieve_from_collection",
        side_effect=_retrieve,
    ):
        for mode in ("ask", "practice", "explain"):
            scope = detect_learning_scope(
                MagicMock(),
                MagicMock(),
                "how do leaves make food using sunlight?",
                agent_mode=mode,
            )
            assert scope.get("chapter_id") == 50, mode
            assert scope.get("chapter_name") == "Chapter 5 - Photosynthesis in Plants", mode


def test_title_stem_overlap_nature_resource():
    score = _title_match_score(
        _norm("when does Nature becomes A resource?"),
        "Chapter 1 - Natural Resources and Their Use",
    )
    assert score >= 2


def test_title_stem_overlap_math_and_science():
    assert (
        _title_match_score(_norm("fraction problems"), "Chapter 2 - Fractions and Decimals")
        >= 2
    )
    assert (
        _title_match_score(
            _norm("how do plants photosynthesize"),
            "Chapter 5 - Photosynthesis in Plants",
        )
        >= 2
    )


def test_title_match_chapter_number_beats_sibling_chapter():
    q = _norm("explain me social chapter 2")
    assert _title_match_score(q, "Chapter 2 - Reshaping India's Political Map") >= 8
    assert _title_match_score(q, "Chapter 1 - Natural Resources and Their Use") < 8


def _run_image_turn(status: str):
    import asyncio

    import app.modules.image_understanding.service as iu
    import app.modules.student_assistant.service as sa
    from app.modules.image_understanding.schemas import ImageUnderstandingResult, UnderstandingBundle

    bundle = UnderstandingBundle(
        result=ImageUnderstandingResult(status=status, content_summary="water cycle"),
        intent="answer",
        retrieval_query="water cycle",
        tutor_prompt_block="IMAGE",
    )

    async def fake_understand(*_a, **_k):
        return bundle

    async def fake_llm(*_a, **_k):
        return "ok"

    scope_calls, rag_calls = MagicMock(return_value={"chapter_id": "1", "subject_name": "Science"}), MagicMock(return_value="")
    with patch.object(iu, "understand", fake_understand), \
         patch.object(sa, "build_student_context", return_value=""), \
         patch.object(sa, "detect_learning_scope", scope_calls), \
         patch.object(sa, "_textbook_snippets", rag_calls), \
         patch.object(sa, "_call_mistral_async", fake_llm), \
         patch.object(sa, "mode_suggested_prompts", return_value=[]), \
         patch.object(sa.learning_service, "get_overview", return_value=SimpleNamespace(class_level="7")):
        asyncio.run(sa.chat(MagicMock(), SimpleNamespace(id=1, full_name="A"), query="answer this", image_ids=["ab"]))
    return scope_calls.call_count, rag_calls.call_count


def test_unclear_image_skips_textbook_search():
    assert _run_image_turn("unclear") == (0, 0)
    assert _run_image_turn("ok") == (1, 1)

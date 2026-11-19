"""Voice stream must emit related_images for textbook turns."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document


_FAKE_IMAGES = [
    {
        "url": "/auth/catalog/textbook-images/u1/fig_2_3.png",
        "caption": "Political map",
        "page": 14,
        "figure_number": "2.3",
    },
    {
        "url": "/auth/catalog/textbook-images/u1/fig_2_12.png",
        "caption": "Political map later",
        "page": 22,
        "figure_number": "2.12",
    },
]


def test_voice_stream_emits_related_images_for_map_question():
    from app.services.chat_service import chapter_aware_qa_stream

    docs = [
        Document(
            page_content="Fig 2.3 shows a political map snapshot for a period.",
            metadata={"page": 13, "_retrieval_score": 0.7},
        ),
        Document(
            page_content="Fig 2.12 and Fig 2.16 are other political map snapshots.",
            metadata={"page": 21, "_retrieval_score": 0.65},
        ),
    ]

    emitted: list[list[dict]] = []

    async def capture(imgs: list[dict]) -> None:
        emitted.append(list(imgs))

    async def fake_llm(_messages, **_kw):
        for tok in ["India's political map changed. ", "Look at the maps on your screen."]:
            yield tok

    async def fake_select(_q, _a, candidates, **_kw):
        return list(candidates or [])

    async def collect() -> tuple[str, list[list[dict]]]:
        parts: list[str] = []
        async for token in chapter_aware_qa_stream(
            "Can you tell me about India's political map?",
            collection_name="CBSE_CLASS_8_Social_Science",
            chapter_ids=["ch-1"],
            chapter="India's political map",
            subject_name="Social Science",
            class_level="CLASS_8",
            board="CBSE",
            voice_mode=True,
            emit_related_images=capture,
        ):
            parts.append(token)
        return "".join(parts), emitted

    with patch("app.services.section_retrieval.retrieve_for_tutor_query", return_value=(docs, MagicMock(kind="general"), "")), patch(
        "app.services.chapter_scope.resolve_chapter_awareness_turn",
        return_value=(None, "Can you tell me about India's political map?", None, None),
    ), patch(
        "app.services.chat_service._stream_mistral_async",
        fake_llm,
    ), patch(
        "app.services.image_service.textbook_image_retrieval.early_related_images_for_query",
        return_value=_FAKE_IMAGES,
    ), patch(
        "app.services.chat_service._fetch_related_images",
        return_value=_FAKE_IMAGES,
    ), patch(
        "app.services.chat_service.qa._select_related_images_for_answer",
        fake_select,
    ):
        answer, batches = asyncio.run(collect())

    assert "political map" in answer.lower()
    assert batches, "expected at least one related_images emit"
    figs = {img.get("figure_number") for batch in batches for img in batch}
    assert "2.3" in figs
    assert "2.12" in figs


def test_llm_image_select_single_final_emit():
    """With LLM select on: one related_images emit after answer (no early)."""
    from app.services.chat_service import chapter_aware_qa_stream

    docs = [
        Document(
            page_content="Fig 2.3 political map.",
            metadata={"page": 13, "_retrieval_score": 0.7},
        ),
    ]
    selected = [_FAKE_IMAGES[0]]
    emitted: list[list[dict]] = []

    async def capture(imgs: list[dict]) -> None:
        emitted.append(list(imgs))

    async def fake_llm(_messages, **_kw):
        yield "India's political map changed over time."

    async def fake_select(question, answer, candidates, **_kw):
        assert "political map" in (question or "").lower() or "map" in (answer or "").lower()
        assert candidates == _FAKE_IMAGES
        return selected

    async def collect() -> list[list[dict]]:
        async for _ in chapter_aware_qa_stream(
            "Can you tell me about India's political map?",
            collection_name="CBSE_CLASS_8_Social_Science",
            chapter_ids=["ch-1"],
            chapter="India's political map",
            subject_name="Social Science",
            class_level="CLASS_8",
            board="CBSE",
            voice_mode=True,
            emit_related_images=capture,
        ):
            pass
        return emitted

    with patch("app.services.chat_service.qa.ENABLE_LLM_IMAGE_SELECT", True), patch(
        "app.services.section_retrieval.retrieve_for_tutor_query",
        return_value=(docs, MagicMock(kind="general"), ""),
    ), patch(
        "app.services.chapter_scope.resolve_chapter_awareness_turn",
        return_value=(None, "Can you tell me about India's political map?", None, None),
    ), patch(
        "app.services.chat_service._stream_mistral_async",
        fake_llm,
    ), patch(
        "app.services.image_service.textbook_image_retrieval.early_related_images_for_query",
        side_effect=AssertionError("early path must not run when LLM select is on"),
    ), patch(
        "app.services.chat_service._fetch_related_images",
        return_value=_FAKE_IMAGES,
    ), patch(
        "app.services.chat_service.qa._select_related_images_for_answer",
        fake_select,
    ):
        batches = asyncio.run(collect())

    assert len(batches) == 1
    assert [b.get("figure_number") for b in batches[0]] == ["2.3"]

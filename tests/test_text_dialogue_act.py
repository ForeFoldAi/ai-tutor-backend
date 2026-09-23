"""Text-to-text LLM-first: closing/ack/intro skip retrieval like voice."""

import asyncio
from unittest.mock import patch

from app.services.chat_service import _resolve_answer_type
from app.services.voice_ack import resolve_dialogue_act


def test_text_dialogue_act_answer_types():
    assert _resolve_answer_type("thanks", dialogue_act="closing") == "affirmation"
    assert _resolve_answer_type("Okay, a nice example.", dialogue_act="ack") == "affirmation"
    assert _resolve_answer_type("I am Seyun", dialogue_act="intro") == "greeting"
    assert resolve_dialogue_act("Good thanks") == "closing"


def test_text_stream_thanks_skips_retrieval():
    from app.services.chat_service import chapter_aware_qa_stream

    def boom(*_a, **_k):
        raise AssertionError("text dialogue act must not retrieve")

    async def fake_llm(messages, **_kw):
        # Compact affirmation path — history-aware, no chapter dump.
        yield "You're welcome! Want a quick quiz or shall we wrap up?"

    history = [
        {"role": "user", "content": "What is social health?"},
        {
            "role": "assistant",
            "content": "Social health means feeling connected with friends and community.",
        },
    ]

    async def collect() -> str:
        out = []
        async for t in chapter_aware_qa_stream(
            "Good thanks",
            collection_name="t",
            voice_mode=False,
            conversation_history=history,
        ):
            out.append(t)
        return "".join(out)

    with patch("app.services.section_retrieval.retrieve_for_tutor_query", boom), patch(
        "app.services.chat_service._stream_mistral_async", fake_llm
    ):
        text = asyncio.run(collect())
    assert "welcome" in text.lower() or "quiz" in text.lower()
    assert "social health means" not in text.lower()


def test_text_content_question_still_retrieves():
    from app.services.chat_service import chapter_aware_qa_stream

    called = {"n": 0}

    def fake_retrieve(*_a, **_k):
        called["n"] += 1
        raise RuntimeError("stop-after-retrieve")

    async def collect() -> None:
        try:
            async for _t in chapter_aware_qa_stream(
                "What is photosynthesis?",
                collection_name="t",
                voice_mode=False,
            ):
                pass
        except RuntimeError as e:
            if "stop-after-retrieve" not in str(e):
                raise

    with patch("app.services.section_retrieval.retrieve_for_tutor_query", fake_retrieve):
        asyncio.run(collect())
    assert called["n"] == 1

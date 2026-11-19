"""Voice dialogue acts skip retrieval; LLM still answers (no canned ack)."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services.voice_ack import (
    is_personal_intro,
    is_voice_acknowledgement,
    resolve_dialogue_act,
)
from app.services.voice_prompts import reply_intent_guidance
from app.services.voice_tutor import ReplyIntent


def test_acks_detected():
    assert is_voice_acknowledgement("(laughing)")
    assert is_voice_acknowledgement("Okay.")
    assert is_voice_acknowledgement("wow")
    assert is_voice_acknowledgement("interesting")
    assert is_voice_acknowledgement("yes")
    assert is_voice_acknowledgement("got it")
    assert is_voice_acknowledgement("Okay, a nice example.")
    assert is_voice_acknowledgement("good explanation")
    assert is_voice_acknowledgement("I understood it very well")
    assert is_voice_acknowledgement("that helped a lot")
    assert is_voice_acknowledgement("makes perfect sense")


def test_educational_requests_are_not_acks():
    assert not is_voice_acknowledgement("Tell me more about the Hoysalas.")
    assert not is_voice_acknowledgement("Did it control all of India?")
    assert not is_voice_acknowledgement("Who established the Mughal Empire?")
    assert not is_voice_acknowledgement("Can you tell me about India's political map?")
    assert not is_voice_acknowledgement("What happened next?")
    assert not is_voice_acknowledgement("go on")
    assert not is_voice_acknowledgement("give me an example")


def test_resolve_dialogue_act():
    assert resolve_dialogue_act("thanks for explaining") == "closing"
    assert resolve_dialogue_act("Okay, a nice example.") == "ack"
    assert resolve_dialogue_act("I am Seyun, D-E-L-H-I.") == "intro"
    assert resolve_dialogue_act("What is weather?") is None
    assert resolve_dialogue_act("x", dialogue_act="closing") == "closing"
    assert is_personal_intro("My name is Suneel")


def test_closing_guidance_forbids_reteach():
    g = reply_intent_guidance(
        ReplyIntent.CLOSING,
        quiz_pending=False,
        quiz_question="",
        quiz_attempts=0,
        dialogue_act="closing",
    ).lower()
    assert "do not repeat" in g or "wrapping up" in g
    assert "welcome back" in g


def test_ack_and_intro_guidance():
    ack = reply_intent_guidance(
        ReplyIntent.NEW_QUESTION,
        quiz_pending=False,
        quiz_question="",
        quiz_attempts=0,
        dialogue_act="ack",
    ).lower()
    assert "do not teach" in ack
    intro = reply_intent_guidance(
        ReplyIntent.NEW_QUESTION,
        quiz_pending=False,
        quiz_question="",
        quiz_attempts=0,
        dialogue_act="intro",
    ).lower()
    assert "introduc" in intro
    assert "not covered" in intro


def test_voice_stream_ack_skips_retrieval_uses_llm():
    from app.services.chat_service import chapter_aware_qa_stream

    def boom(*_a, **_k):
        raise AssertionError("ack must not retrieve")

    async def fake_llm(messages, **_kw):
        yield "Glad that helped — want a quick quiz next?"

    async def collect(query: str, **kw) -> str:
        out = []
        async for t in chapter_aware_qa_stream(
            query, collection_name="t", voice_mode=True, **kw
        ):
            out.append(t)
        return "".join(out)

    with patch("app.services.section_retrieval.retrieve_for_tutor_query", boom), patch(
        "app.services.chat_service._stream_mistral_async", fake_llm
    ), patch(
        "app.services.student_affect.evaluate_student_affect_async",
        AsyncMock(
            return_value=__import__(
                "app.services.student_affect", fromlist=["StudentAffect"]
            ).StudentAffect(primary="affirmation", is_affirmation=True)
        ),
    ):
        okay = asyncio.run(collect("Okay."))
        thanks = asyncio.run(collect("thanks for explaining", dialogue_act="closing"))
    assert "Glad" in okay or "quiz" in okay.lower() or len(okay) > 5
    assert len(thanks) > 5
    assert "Alright." != okay  # no canned short-circuit


def test_quiz_pending_yes_still_retrieves():
    from app.services.chat_service import chapter_aware_qa_stream
    from app.services.student_affect import StudentAffect

    called = {"n": 0}

    def fake_retrieve(*_a, **_k):
        called["n"] += 1
        raise RuntimeError("stop-after-retrieve")

    async def fake_affect(*_a, **_k):
        return StudentAffect(primary="affirmation", is_affirmation=True)

    async def collect() -> None:
        try:
            async for _t in chapter_aware_qa_stream(
                "yes",
                collection_name="t",
                voice_mode=True,
                quiz_pending=True,
            ):
                pass
        except RuntimeError as e:
            if "stop-after-retrieve" not in str(e):
                raise

    with patch(
        "app.services.student_affect.evaluate_student_affect_async", fake_affect
    ), patch(
        "app.services.section_retrieval.retrieve_for_tutor_query", fake_retrieve
    ):
        asyncio.run(collect())
    assert called["n"] == 1

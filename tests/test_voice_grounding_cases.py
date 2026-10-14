"""Required production cases: grounding + follow-up + live LLM when available."""

import asyncio
import os

from app.services.conversation_context import resolve_conversation_context
from app.services.voice_ack import is_voice_acknowledgement, voice_ack_reply
from app.services.voice_tutor import TutorState, build_voice_mistral_messages
from app.services.section_retrieval import query_breadth

MAP_CONTEXT = """
The political map of India changed many times during this period.
The Delhi Sultanate controlled large parts of northern India, but not the entire subcontinent.
Several regional kingdoms remained independent.
The Vijayanagara Empire was an important power in the south.
The Ahom Kingdom ruled in the north-east.
Babur established the Mughal Empire after the First Battle of Panipat.
The Hoysalas ruled in the region of Karnataka.
"""

HOYSALA_CONTEXT = """
The Hoysalas were a regional kingdom in southern India.
They ruled in the region of Karnataka.
They resisted the expansion of the Delhi Sultanate.
"""

THIN_CONTEXT = """
Some regional kingdoms resisted the expansion of the Delhi Sultanate.
"""


def _collect_llm(query: str, context: str) -> str:
    from app.services.chat_service import _stream_mistral_async

    messages = build_voice_mistral_messages(
        query,
        context,
        class_level="CLASS_7",
        board="CBSE",
        subject_name="Social Science",
        chapter="India's political map",
        tutor_state=TutorState.TEACHING,
    )

    async def run() -> str:
        parts: list[str] = []
        async for token in _stream_mistral_async(messages, feature="voice"):
            parts.append(token)
        return "".join(parts)

    return asyncio.run(run())


def test_1_broad_question_is_broad_retrieval():
    q = "Can you tell me about India's political map?"
    assert query_breadth(q) == "broad"
    assert not is_voice_acknowledgement(q)


def test_2_laughing_is_brief_ack():
    q = "(laughing)"
    assert is_voice_acknowledgement(q)
    out = voice_ack_reply(q)
    assert "enjoying" in out.lower()
    assert "ahom" not in out.lower()
    assert "river" not in out.lower()


def test_3_simple_fact_not_an_ack():
    q = "Who established the Mughal Empire?"
    assert not is_voice_acknowledgement(q)


def test_4_follow_up_resolves_it_to_delhi_sultanate():
    ctx = resolve_conversation_context(
        "Did it control all of India?",
        conversation_history=[
            {"role": "user", "content": "What was the Delhi Sultanate?"},
            {
                "role": "assistant",
                "content": "The Delhi Sultanate was a series of dynasties in north India.",
            },
        ],
        chapter="India's political map",
    )
    blob = f"{ctx.retrieval_query} {ctx.resolved_topic}".lower()
    assert "delhi sultanate" in blob
    assert not is_voice_acknowledgement("Did it control all of India?")


def test_5_unsupported_detail_prompt_forbids_guessing():
    from app.services.voice_prompts import build_voice_user_message, VOICE_SYSTEM_PROMPT

    user = build_voice_user_message(
        "What did Ram Singh himself say about the Ahoms?", THIN_CONTEXT
    )
    assert "do not invent" in user.lower()
    assert "general knowledge" not in user.lower()
    assert "Never invent a quotation" in VOICE_SYSTEM_PROMPT


def test_6_haldighati_not_in_retrieved_context():
    assert "haldighati" not in MAP_CONTEXT.lower()


def test_7_okay_is_brief_ack():
    q = "Okay."
    assert is_voice_acknowledgement(q)
    assert voice_ack_reply(q) == "Alright."


def test_8_explicit_continuation_is_not_an_ack():
    q = "Tell me more about the Hoysalas."
    assert not is_voice_acknowledgement(q)


def test_live_llm_grounding_against_textbook_context():
    """Actual spoken answers from the voice prompt + Mistral, given fixed context."""
    from app.config import MISTRAL_API_KEY

    if not (MISTRAL_API_KEY or "").strip():
        raise AssertionError("MISTRAL_API_KEY missing — cannot run live grounding cases")

    results = {}

    results["t1"] = _collect_llm(
        "Can you tell me about India's political map?", MAP_CONTEXT
    )
    results["t3"] = _collect_llm("Who established the Mughal Empire?", MAP_CONTEXT)
    results["t5"] = _collect_llm(
        "What did Ram Singh himself say about the Ahoms?", THIN_CONTEXT
    )
    results["t6"] = results["t1"]
    results["t8"] = _collect_llm("Tell me more about the Hoysalas.", HOYSALA_CONTEXT)

    print("\n=== LIVE VOICE OUTPUT ===")
    for k, v in results.items():
        print(f"\n[{k}]\n{v}\n")

    t1 = results["t1"].lower()
    assert "haldighati" not in t1
    assert "imagine you're looking" not in t1
    assert any(w in t1 for w in ("delhi", "vijayanagara", "ahom", "mughal", "kingdom"))

    t3 = results["t3"].lower()
    assert "babur" in t3
    assert "haldighati" not in t3

    t5 = results["t5"].lower()
    assert "don't see" in t5 or "does not" in t5 or "don't have" in t5 or "not enough" in t5 or "textbook" in t5
    assert "ram singh himself said" not in t5
    assert "haldighati" not in t5

    t8 = results["t8"].lower()
    assert "hoysala" in t8
    assert "karnataka" in t8 or "south" in t8 or "resist" in t8
    assert "haldighati" not in t8
    assert "belur" not in t8
    assert "halebid" not in t8

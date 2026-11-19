"""Voice tutor accuracy: homophone repair, history hygiene, quiz hijack, provider failover."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.services.voice_chapter_vocab import (
    build_vocab_from_texts,
    is_chapter_overview_request,
    is_student_correction,
)
from app.services.voice_tutor import ReplyIntent, classify_reply_intent

CHAPTER = [
    "Tricki was a small dog. Mrs Pumphrey loved Tricki and fed Tricki too much. "
    "The vet came to see Tricki. The vet told Mrs Pumphrey to stop. Tricki and the vet became friends. "
    "The vet gave Tricki medicine and the vet smiled. Mrs Pumphrey cried. Tricki was wet with tears once."
]


@pytest.fixture()
def vocab():
    return build_vocab_from_texts(CHAPTER, ["1. A Triumph of Surgery"])


def test_dog_name_beats_tricky(vocab):
    text, _ = vocab.resolve("Who is tricky?")
    assert "Tricki" in text


def test_vet_vs_wet(vocab):
    text, _ = vocab.resolve("Who is wet?")
    assert "vet" in text.lower() and "wet" not in text.lower()


def test_common_words_not_rewritten(vocab):
    text, _ = vocab.resolve("What is the story about?")
    assert text == "What is the story about?"


def test_relevance_flags_off_topic(vocab):
    assert vocab.relevance("What did the vet do for Tricki?") > 0.5
    assert vocab.relevance("pump red kinetic energy") < 0.3


def test_correction_and_overview_intents():
    assert is_student_correction("I am not saying pricky I am saying tricky")
    assert not is_student_correction("What did the vet do?")
    assert is_chapter_overview_request("explain this chapter in a simple way starting to ending")
    assert is_chapter_overview_request("What is the story all about?")


def test_new_question_is_not_a_quiz_answer():
    q = "But here we are talking about Tricki so why did pushing come in?"
    assert classify_reply_intent(q, quiz_pending=True) == ReplyIntent.NEW_QUESTION
    assert classify_reply_intent("it will move faster", quiz_pending=True) == ReplyIntent.WRONG_ANSWER


def test_retract_last_turn_drops_pair_and_quiz_state():
    from app.voice_ws import _Session, _retract_last_turn

    s = _Session()
    s.history = [
        {"role": "user", "content": "good one"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Who is tricky?"},
        {"role": "assistant", "content": "Mrs Pumphrey!"},
    ]
    s.quiz_pending, s.quiz_question, s.quiz_attempts = True, "what?", 1
    _retract_last_turn(s)
    assert [t["content"] for t in s.history] == ["good one", "ok"]
    assert not s.quiz_pending and s.quiz_attempts == 0


def test_refine_flags_doubtful_offtopic_turn(vocab):
    from app.voice_ws import _Session, _refine_transcript

    s = _Session()
    s.vocab = vocab
    out = asyncio.run(_refine_transcript(s, "pump red kinetic energy pushing", None))
    assert out["suspect"] is True  # off-chapter, not a follow-up -> kept out of history
    ok = asyncio.run(_refine_transcript(s, "who is tricky", None))
    assert "Tricki" in ok["text"] and ok["suspect"] is False


def test_failover_to_second_provider_on_429(monkeypatch):
    from app.services import llm_client as c

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "primary.test" in str(request.url):
            return httpx.Response(429, headers={"Retry-After": "30"}, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi there"}}]})

    targets = [
        {"name": "p", "base_url": "https://primary.test/v1", "key": "k", "model": "m1"},
        {"name": "s", "base_url": "https://second.test/v1", "key": "k", "model": "m2"},
    ]
    monkeypatch.setattr(c, "_targets", lambda feature: targets)
    monkeypatch.setattr(c, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(c, "ensure_llm_config", lambda feature="chat": None)
    c._loop_state.clear()

    async def run():
        return await c.complete([{"role": "user", "content": "x"}], feature="chat")

    assert asyncio.run(run()) == "hi there"
    assert len(calls) == 2 and "second.test" in calls[1]


def test_groq_keys_rotate_then_fall_to_secondary(monkeypatch):
    from app.services import key_pool, llm_client as c

    key_pool.reset()
    monkeypatch.setenv("GROQ_API_KEY", "g1")
    monkeypatch.setenv("GROQ_API_KEYS", "g2, g3\ng1")
    from app.config import provider_keys

    assert provider_keys("groq") == ["g1", "g2", "g3"]  # de-duplicated, ordered

    seen: list[str] = []
    busy = {"g1", "g2", "g3"}

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"].removeprefix("Bearer ")
        seen.append(auth)
        if auth in busy:
            return httpx.Response(429, headers={"Retry-After": "30"}, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(c, "llm_base_url_for", lambda feature="chat": "https://groq.test/v1")
    monkeypatch.setattr(c, "llm_key_pool_for", lambda feature="chat": ["g1", "g2", "g3"])
    monkeypatch.setattr(c, "llm_model_for", lambda feature="chat": "m")
    monkeypatch.setattr(
        c,
        "llm_fallback_for",
        lambda feature="chat": {"base_url": "https://mistral.test/v1", "key": "m1", "keys": ["m1"], "model": "m"},
    )
    targets = c._targets

    monkeypatch.setattr(c, "_targets", targets)
    monkeypatch.setattr(c, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(c, "ensure_llm_config", lambda feature="chat": None)
    c._loop_state.clear()

    async def ask():
        return await c.complete([{"role": "user", "content": "x"}], feature="chat")

    # All three Groq keys busy -> each tried once, then the secondary answers.
    assert asyncio.run(ask()) == "ok"
    assert sorted(seen[:3]) == ["g1", "g2", "g3"] and seen[3] == "m1"

    # Keys are now parked: the next call goes straight to the secondary, no wasted tries.
    seen.clear()
    assert asyncio.run(ask()) == "ok"
    assert seen == ["m1"]

    # One key recovers -> traffic returns to Groq (usable keys sort first).
    busy.discard("g2")
    key_pool.reset()
    seen.clear()
    assert asyncio.run(ask()) == "ok"
    assert seen[-1] in {"g2"} and "m1" not in seen

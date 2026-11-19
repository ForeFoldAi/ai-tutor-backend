"""All Groq keys get their chance before the secondary provider (Mistral) answers."""

import asyncio

import httpx
import pytest

from app.main import app  # noqa: F401 — app import order avoids circular imports
from app.services import key_pool, llm_client as c

GROQ = "https://groq.test/v1"
KEYS = ["g1", "g2", "g3", "g4", "g5"]


@pytest.fixture
def llm(monkeypatch):
    key_pool.reset()
    seen: list[str] = []
    replies: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"].removeprefix("Bearer ")
        seen.append(auth)
        reply = replies.get(auth, 200)
        if isinstance(reply, Exception):
            raise reply
        if reply == 200:
            return httpx.Response(200, json={"choices": [{"message": {"content": f"from {auth}"}}]})
        return httpx.Response(reply, headers={"Retry-After": "30"}, json={})

    monkeypatch.setattr(c, "llm_base_url_for", lambda feature="chat": GROQ)
    monkeypatch.setattr(c, "llm_key_pool_for", lambda feature="chat": list(KEYS))
    monkeypatch.setattr(c, "llm_model_for", lambda feature="chat": "m")
    monkeypatch.setattr(
        c,
        "llm_fallback_for",
        lambda feature="chat": {"base_url": "https://mistral.test/v1", "key": "m1", "keys": ["m1"], "model": "m"},
    )
    monkeypatch.setattr(c, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(c, "ensure_llm_config", lambda feature="chat": None)
    c._loop_state.clear()

    def ask() -> str:
        return asyncio.run(c.complete([{"role": "user", "content": "x"}], feature="assistant"))

    yield seen, replies, ask
    key_pool.reset()


def test_rate_limited_groq_keys_are_all_tried_before_mistral(llm):
    seen, replies, ask = llm
    replies.update({k: 429 for k in KEYS})
    assert ask() == "from m1"
    assert sorted(seen[:5]) == KEYS and seen[5] == "m1"


def test_revoked_groq_key_moves_to_next_groq_key(llm):
    seen, replies, ask = llm
    replies.update({k: 401 for k in KEYS[:4]})
    assert ask() == "from g5"
    assert "m1" not in seen


def test_groq_outage_skips_remaining_groq_keys(llm):
    seen, replies, ask = llm
    replies.update({k: httpx.ConnectError("down") for k in KEYS})
    assert ask() == "from m1"
    assert seen[-1] == "m1" and len(seen) == 2


def test_request_too_large_for_groq_goes_to_mistral(llm):
    seen, replies, ask = llm
    replies.update({k: 413 for k in KEYS})
    assert ask() == "from m1"
    assert len(seen) == 2


def test_groq_key_free_within_a_moment_beats_mistral(llm):
    seen, replies, ask = llm
    for k in KEYS:
        key_pool.cool(GROQ, k, 0.5 if k == "g3" else 30.0)
    targets = [t["key"] for t in c._targets("assistant")]
    assert targets[0] == "g3" and targets[1] == "m1"

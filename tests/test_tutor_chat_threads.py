"""Multi-thread tutor chat archive (New Chat keeps prior threads)."""

from __future__ import annotations

from app.modules.student_learning.schemas import TutorChatMessageIn, TutorChatPutRequest
from app.modules.student_learning.tutor_chat_storage import parse_chat_storage, serialize_chat_storage


def test_legacy_flat_messages_migrate_to_single_thread():
    doc = parse_chat_storage(
        [
            {"id": "1", "role": "user", "content": "What is weather?"},
            {"id": "2", "role": "assistant", "content": "Weather is..."},
        ]
    )
    assert len(doc["threads"]) == 1
    assert doc["threads"][0]["title"].startswith("What is weather")
    assert len(doc["threads"][0]["messages"]) == 2


def test_v2_roundtrip_keeps_threads():
    raw = serialize_chat_storage(
        {
            "active_thread_id": "t2",
            "threads": [
                {
                    "id": "t1",
                    "title": "First",
                    "messages": [{"id": "1", "role": "user", "content": "Hi"}],
                    "updated_at": "2026-01-01T00:00:00Z",
                },
                {
                    "id": "t2",
                    "title": "Second",
                    "messages": [{"id": "2", "role": "user", "content": "Bye"}],
                    "updated_at": "2026-01-02T00:00:00Z",
                },
            ],
        }
    )
    doc = parse_chat_storage(raw)
    assert doc["active_thread_id"] == "t2"
    assert [t["id"] for t in doc["threads"]] == ["t1", "t2"]


def test_put_schema_accepts_thread_fields() -> None:
    payload = TutorChatPutRequest(
        subject_name="Social",
        messages=[TutorChatMessageIn(id="1", role="user", content="Rain?")],
        thread_id="thread-abc",
        active_thread_id="thread-abc",
        new_thread=True,
    )
    assert payload.new_thread is True
    assert payload.thread_id == "thread-abc"

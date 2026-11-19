"""Tutor chat message sanitization (no DB)."""

from __future__ import annotations

from app.modules.student_learning.schemas import TutorChatMessageIn, TutorChatPutRequest


def test_tutor_chat_put_schema_accepts_turns() -> None:
    payload = TutorChatPutRequest(
        subject_name="Science",
        messages=[
            TutorChatMessageIn(id="1", role="user", content="What is climate?"),
            TutorChatMessageIn(id="2", role="assistant", content="Climate is..."),
        ],
    )
    assert len(payload.messages) == 2
    assert payload.messages[0].role == "user"

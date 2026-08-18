"""Rolling conversation memory for long tutor sessions."""

from app.services.conversation_memory import (
    ConversationMemory,
    format_memory_for_prompt,
    prepare_conversation_inputs,
    remember_turn,
    update_memory_after_turn,
)


def _turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_prepare_conversation_inputs_folds_old_turns():
    history = [
        _turn("user", f"Question about topic {i}")
        for i in range(30)
        for _ in (0, 1)
    ]
    # flatten to user/assistant pairs
    paired: list[dict] = []
    for i in range(20):
        paired.append(_turn("user", f"Explain concept {i}"))
        paired.append(_turn("assistant", f"Concept {i} is about history and geography."))

    recent, mem = prepare_conversation_inputs(paired, chapter="Chapter 3")
    assert len(recent) <= 16
    assert mem.turn_count > 0 or mem.conversation_summary
    assert "Concept 0" in mem.conversation_summary or mem.turn_count >= 1


def test_remember_turn_folds_when_history_cap_exceeded():
    mem = ConversationMemory()
    history: list[dict] = []
    for i in range(12):
        history, mem = remember_turn(
            history,
            mem,
            user_query=f"Why did event {i} happen?",
            assistant_response=f"Event {i} happened because of politics.",
            resolved_topic=f"event {i}",
            chapter="Chapter 3",
            history_cap=20,
        )
    assert len(history) <= 20
    assert mem.turn_count == 12
    assert mem.last_student_question.startswith("Why did event 11")


def test_format_memory_for_prompt_includes_summary():
    mem = ConversationMemory(
        conversation_summary="Student asked about Shivaji; tutor explained successors.",
        topics_discussed=["Chapter 3 - Marathas", "Shivaji"],
        current_concept="Mughal attacks",
        last_student_question="Why?",
        turn_count=3,
    )
    block = format_memory_for_prompt(mem)
    assert "SESSION MEMORY" in block
    assert "Shivaji" in block
    assert "Why?" in block


def test_update_memory_after_turn_tracks_concept():
    mem = update_memory_after_turn(
        ConversationMemory(),
        user_query="Can you summarize the key points?",
        assistant_response="Key points include Shivaji and Sambhaji.",
        resolved_topic="Shivaji successors",
        followup_type="ask_summary",
        chapter="Chapter 3",
    )
    assert mem.turn_count == 1
    assert "summarize" in mem.last_student_question.lower()
    assert mem.current_concept == "Shivaji successors"


def test_500_turn_session_no_crash():
    mem = ConversationMemory()
    history: list[dict] = []
    for i in range(520):
        history, mem = remember_turn(
            history,
            mem,
            user_query="Why?" if i % 2 == 0 else "Can you explain that again?",
            assistant_response=f"Explanation part {i} about the Maratha kingdom.",
            resolved_topic="Maratha kingdom",
            chapter="Chapter 3",
            history_cap=20,
        )
    assert len(history) <= 20
    assert mem.turn_count == 520
    assert len(mem.conversation_summary) <= 3500
    block = format_memory_for_prompt(mem)
    assert block
    recent, rebuilt = prepare_conversation_inputs(history, memory=mem, chapter="Chapter 3")
    assert len(recent) <= 16

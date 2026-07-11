"""Phase 1 — voice conversation turn lifecycle."""

from app.services.voice_conversation import emit_interrupt_listening
from app.services.voice_tutor import TutorState


def test_emit_interrupt_listening_payload_order():
    sent: list[dict] = []

    async def capture(_ws, payload: dict) -> None:
        sent.append(payload)

    import asyncio

    asyncio.run(emit_interrupt_listening(capture, None))  # type: ignore[arg-type]

    assert sent[0]["type"] == "interrupt_ack"
    assert sent[1]["type"] == "listening"
    assert sent[2]["type"] == "tutor_state"
    assert sent[2]["state"] == TutorState.LISTENING.value

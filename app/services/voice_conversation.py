"""
Voice conversation turn lifecycle helpers (Phase 1).

Keeps interrupt / listening transitions explicit and race-free without
changing the WebSocket message schema.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from fastapi import WebSocket

from app.services.voice_tutor import TutorState

SendFn = Callable[[WebSocket, dict], Awaitable[None]]


async def emit_interrupt_listening(
    send: SendFn,
    ws: WebSocket,
) -> None:
    """
  Immediate client feedback on barge-in:
  1. interrupt_ack — stop UI "speaking"; client commits partial assistant text then clears the live stream
  2. listening — mic may open; client restarts capture
  3. tutor_state — server session returns to LISTENING

  Generation cancel runs after ack so the user hears silence instantly.
  """
    await send(ws, {"type": "interrupt_ack"})
    await send(ws, {"type": "listening"})
    await send(ws, {"type": "tutor_state", "state": TutorState.LISTENING.value})

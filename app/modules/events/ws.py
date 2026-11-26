from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.security import decode_token
from app.modules.auth.single_session import WS_CLOSE_SESSION_REPLACED, sid_is_current
from app.modules.events.hub import hub
from app.modules.users.models import User

logger = logging.getLogger(__name__)

events_ws_router = APIRouter()


def _authenticate_ws_token(token: str | None) -> tuple[int, str | None] | None:
    """(user id, session id) from a valid access token, else None."""
    if not token:
        return None
    try:
        payload = decode_token(token)
    except ValueError:
        return None
    if payload.get("type") != "access":
        return None
    try:
        return int(payload.get("sub")), payload.get("sid")
    except (TypeError, ValueError):
        return None


def _session_replaced(user_id: int, sid: str | None) -> bool:
    """True when this student has since signed in on another device."""
    db: Session = SessionLocal()
    try:
        user = db.get(User, user_id)
        return user is not None and not sid_is_current(user, sid)
    finally:
        db.close()


def _rooms_for_user(user: User) -> list[str]:
    rooms = ["global"]
    if user.school_id is not None:
        rooms.append(f"school:{user.school_id}")
    rooms.append(f"user:{user.id}")
    return rooms


async def _send_json(ws: WebSocket, payload: dict) -> None:
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))


@events_ws_router.websocket("/ws/events")
async def domain_events_ws(
    ws: WebSocket,
    access_token: str | None = Query(None),
) -> None:
    # Accept first so the browser gets a proper handshake (close-before-accept
    # surfaces as "Connection closed before receiving a handshake response").
    await ws.accept()

    auth = _authenticate_ws_token(access_token)
    if not auth:
        await ws.close(code=4401)
        return
    user_id, sid = auth

    db: Session = SessionLocal()
    try:
        user = db.get(User, user_id)
        if user is None or not user.is_active:
            await ws.close(code=4403)
            return
        if not sid_is_current(user, sid):
            await ws.close(code=WS_CLOSE_SESSION_REPLACED)
            return
        rooms = _rooms_for_user(user)
    finally:
        db.close()

    queue = await hub.subscribe(rooms)
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            if await asyncio.to_thread(_session_replaced, user_id, sid):
                stop.set()
                await _send_json(ws, {"type": "session_replaced"})
                await ws.close(code=WS_CLOSE_SESSION_REPLACED)
                return
            await _send_json(ws, {"type": "heartbeat"})
            try:
                await asyncio.wait_for(stop.wait(), timeout=25.0)
            except asyncio.TimeoutError:
                pass

    hb_task = asyncio.create_task(heartbeat())
    await _send_json(ws, {"type": "connected", "rooms": rooms})

    try:
        while True:
            getter = asyncio.create_task(queue.get())
            recv = asyncio.create_task(ws.receive_text())
            done, pending = await asyncio.wait(
                {getter, recv},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if getter in done:
                event = getter.result()
                await _send_json(ws, event)
            if recv in done:
                # Client ping / ignore payload; disconnect raises.
                _ = recv.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("domain events ws error user=%s", user_id)
    finally:
        stop.set()
        hb_task.cancel()
        await hub.unsubscribe(rooms, queue)

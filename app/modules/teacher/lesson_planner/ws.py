from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.core.database import SessionLocal
from app.core.security import decode_token
from app.modules.teacher.lesson_planner.constants import WS_HEARTBEAT
from app.modules.teacher.lesson_planner.models import LessonJob
from app.services.lesson_planner.redis.job_state import get_job_state, get_sync_redis, pubsub_channel

logger = logging.getLogger(__name__)

lesson_planner_ws_router = APIRouter()


def _authenticate_ws_token(token: str | None) -> int | None:
    if not token:
        return None
    try:
        payload = decode_token(token)
    except ValueError:
        return None
    if payload.get("type") != "access":
        return None
    try:
        return int(payload.get("sub"))
    except (TypeError, ValueError):
        return None


async def _send_json(ws: WebSocket, payload: dict) -> None:
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))


@lesson_planner_ws_router.websocket("/ws/lesson-planner/{job_id}")
async def lesson_planner_job_ws(
    ws: WebSocket,
    job_id: int,
    access_token: str | None = Query(None),
) -> None:
    user_id = _authenticate_ws_token(access_token)
    if not user_id:
        await ws.close(code=4401)
        return

    db = SessionLocal()
    try:
        job = db.get(LessonJob, job_id)
        if not job or job.user_id != user_id:
            await ws.close(code=4403)
            return
    finally:
        db.close()

    await ws.accept()

    cached = get_job_state(str(job_id))
    if cached:
        for event in cached.get("events") or []:
            await _send_json(ws, event)

    redis = get_sync_redis()
    pubsub = redis.pubsub(ignore_subscribe_messages=True)
    channel = pubsub_channel(str(job_id))
    pubsub.subscribe(channel)

    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            await _send_json(ws, {"event": WS_HEARTBEAT})
            try:
                await asyncio.wait_for(stop.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                continue

    async def listen_redis() -> None:
        while not stop.is_set():
            message = await asyncio.to_thread(pubsub.get_message, timeout=1.0)
            if not message or message.get("type") != "message":
                continue
            try:
                event = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            await _send_json(ws, event)
            if event.get("event") in {"completed", "cancelled", "error"}:
                stop.set()
                break

    heartbeat_task = asyncio.create_task(heartbeat())
    listener_task = asyncio.create_task(listen_redis())

    try:
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ping":
                await _send_json(ws, {"event": "pong"})
    finally:
        stop.set()
        heartbeat_task.cancel()
        listener_task.cancel()
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
        except Exception:
            pass

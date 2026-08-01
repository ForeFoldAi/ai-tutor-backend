from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DomainEvent:
    entity: str
    action: str
    school_id: int | None
    actor_id: int | None
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "domain_event",
            "entity": self.entity,
            "action": self.action,
            "school_id": self.school_id,
            "actor_id": self.actor_id,
            "payload": self.payload or {},
        }


class DomainEventHub:
    """In-process fanout for /ws/events. Rooms: school:{id} | global | user:{id}."""

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, rooms: list[str]) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        async with self._lock:
            for room in rooms:
                self._subs[room].add(q)
        return q

    async def unsubscribe(self, rooms: list[str], q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            for room in rooms:
                self._subs[room].discard(q)
                if not self._subs[room]:
                    self._subs.pop(room, None)

    async def publish(self, event: DomainEvent) -> None:
        rooms = {"global"}
        if event.school_id is not None:
            rooms.add(f"school:{event.school_id}")
        # Targeted fanout for inbox rows (clients subscribe to user:{id}).
        for rid in (event.payload or {}).get("recipient_ids") or ():
            try:
                rooms.add(f"user:{int(rid)}")
            except (TypeError, ValueError):
                continue
        data = event.to_dict()
        async with self._lock:
            targets: set[asyncio.Queue[dict[str, Any]]] = set()
            for room in rooms:
                targets |= self._subs.get(room, set())
        for q in targets:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                # ponytail: drop oldest-style — skip if slow client; they refetch on focus
                logger.debug("domain event dropped for slow subscriber: %s", event.entity)


hub = DomainEventHub()


def publish_domain_event(
    *,
    entity: str,
    action: str,
    school_id: int | None,
    actor_id: int | None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget from sync FastAPI routes."""
    event = DomainEvent(
        entity=entity,
        action=action,
        school_id=school_id,
        actor_id=actor_id,
        payload=payload,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop (rare sync context) — skip; clients still get focus refetch.
        logger.debug("no event loop for domain event %s", entity)
        return
    loop.create_task(hub.publish(event))

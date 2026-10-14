"""Persisted tutor chat thread document (JSONB in student_tutor_chats.messages)."""

from __future__ import annotations

import uuid

_CHAT_MESSAGE_CAP = 80
_CHAT_THREAD_CAP = 12


def sanitize_chat_messages(raw: list) -> list[dict]:
    cleaned: list[dict] = []
    for m in raw[-_CHAT_MESSAGE_CAP:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        mid = m.get("id")
        if role not in ("user", "assistant") or not isinstance(content, str) or not mid:
            continue
        text = content.strip()
        if not text:
            continue
        cleaned.append(
            {
                "id": str(mid),
                "role": role,
                "content": text[:20000],
                "created_at": m.get("created_at") if isinstance(m.get("created_at"), str) else None,
            }
        )
    return cleaned


def thread_title(messages: list[dict], *, fallback: str = "Chat") -> str:
    for m in messages:
        if m.get("role") == "user":
            text = str(m.get("content") or "").strip()
            if text:
                return (text[:48] + "…") if len(text) > 48 else text
    return fallback


def new_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex[:12]}"


def parse_chat_storage(raw) -> dict:
    """Normalize legacy flat message list and v2 thread documents."""
    if isinstance(raw, dict) and raw.get("v") == 2 and isinstance(raw.get("threads"), list):
        threads: list[dict] = []
        for t in raw["threads"]:
            if not isinstance(t, dict) or not t.get("id"):
                continue
            msgs = sanitize_chat_messages(t.get("messages") or [])
            threads.append(
                {
                    "id": str(t["id"]),
                    "title": str(t.get("title") or thread_title(msgs)),
                    "messages": msgs,
                    "updated_at": t.get("updated_at"),
                }
            )
        active = str(raw.get("active_thread_id") or "")
        if threads and not any(t["id"] == active for t in threads):
            active = threads[-1]["id"]
        if not threads:
            active = ""
        return {"active_thread_id": active, "threads": threads}

    legacy_msgs = sanitize_chat_messages(raw if isinstance(raw, list) else [])
    if not legacy_msgs:
        return {"active_thread_id": "", "threads": []}
    tid = f"thread-legacy-{legacy_msgs[0]['id']}"
    return {
        "active_thread_id": tid,
        "threads": [
            {
                "id": tid,
                "title": thread_title(legacy_msgs),
                "messages": legacy_msgs,
                "updated_at": None,
            }
        ],
    }


def serialize_chat_storage(doc: dict) -> dict:
    return {
        "v": 2,
        "active_thread_id": doc.get("active_thread_id") or "",
        "threads": doc.get("threads") or [],
    }


def cap_threads(doc: dict, *, active: str) -> None:
    doc["threads"] = [t for t in doc["threads"] if t.get("messages") or t.get("id") == active]
    doc["threads"].sort(key=lambda t: t.get("updated_at") or "")
    if len(doc["threads"]) <= _CHAT_THREAD_CAP:
        return
    overflow = [t for t in doc["threads"] if t.get("id") != active]
    keep = overflow[-(_CHAT_THREAD_CAP - 1) :]
    active_rows = [t for t in doc["threads"] if t.get("id") == active]
    doc["threads"] = keep + active_rows
    doc["threads"].sort(key=lambda t: t.get("updated_at") or "")

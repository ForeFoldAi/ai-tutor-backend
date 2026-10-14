"""ponytail: assert-based self-check for notification schemas + path map (no DB)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.modules.events.middleware import _entity_for_path
from app.modules.notifications.schemas import (
    MasterNotifyRequest,
    NotificationOut,
    MarkReadRequest,
)


def main() -> None:
    assert _entity_for_path("/auth/me/notifications") == "notifications"
    assert _entity_for_path("/auth/me/notifications/mark-read") == "notifications"
    assert _entity_for_path("/auth/master/notifications") == "notifications"

    # Recipient dedupe (same rule as notify_users).
    ids = list(dict.fromkeys(int(i) for i in [3, 1, 3, 2, 1] if i is not None))
    assert ids == [3, 1, 2]

    now = datetime.now(UTC)
    out = NotificationOut(
        id=1,
        type="teacher_created",
        title="Welcome",
        body="You were added",
        link="/tutor/dashboard",
        read_at=None,
        created_at=now,
        actor_user_id=9,
    )
    assert out.title == "Welcome" and out.read_at is None

    req = MasterNotifyRequest(
        recipient_ids=[1, 2],
        title="Hello",
        body="Please review enrollment.",
    )
    assert req.title == "Hello" and len(req.recipient_ids) == 2

    mark_all = MarkReadRequest(ids=None)
    assert mark_all.ids is None
    mark_some = MarkReadRequest(ids=[1, 2])
    assert mark_some.ids == [1, 2]

    print("notifications self-check ok")


if __name__ == "__main__":
    main()

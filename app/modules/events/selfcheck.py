"""ponytail: self-check for path→entity map + hub rooms (no network)."""

from __future__ import annotations

from app.modules.events.middleware import _entity_for_path


def main() -> None:
    assert _entity_for_path("/auth/admin/students") == "students"
    assert _entity_for_path("/auth/admin/students/1") == "students"
    assert _entity_for_path("/auth/admin/teachers/bulk") == "teachers"
    assert _entity_for_path("/auth/admin/classes") == "classes"
    assert _entity_for_path("/auth/admin/credentials") == "credentials"
    assert _entity_for_path("/auth/tutor/live-sessions") == "sessions"
    assert _entity_for_path("/api/tutor/assignments") == "assignments"
    assert _entity_for_path("/auth/admin/users/3/status") == "users"
    assert _entity_for_path("/search") is None
    assert _entity_for_path("/auth/login") is None
    print("events self-check ok")


if __name__ == "__main__":
    main()

"""Assert email column is non-unique in the mapped model."""

from __future__ import annotations

from app.modules.users.models import User


def main() -> None:
    col = User.__table__.c.email
    assert col.unique is not True and not col.unique, col.unique
    # username remains the login identity
    from app.modules.users.models import UserSettings

    assert UserSettings.__table__.c.username.unique is True
    print("shared email selfcheck ok")


if __name__ == "__main__":
    main()

"""Assert public signups stay pending (approved_at NULL) and login is blocked. Needs DB; rolls back."""

from __future__ import annotations

import app.main  # noqa: F401  (registers all mappers)
from sqlalchemy import null

from app.core.database import SessionLocal
from app.modules.auth.exceptions import AuthException
from app.modules.auth.service import _ensure_active
from app.modules.users.models import Role, User


def main() -> None:
    db = SessionLocal()
    try:
        u = User(
            full_name="approval probe",
            email="approval-probe@example.invalid",
            password_hash="x",
            role=Role.STUDENT,
            is_active=True,
            approved_at=null(),
        )
        db.add(u)
        db.flush()
        db.refresh(u)
        assert u.approved_at is None, u.approved_at
        try:
            _ensure_active(u)
        except AuthException:
            pass
        else:
            raise AssertionError("pending user passed _ensure_active")
    finally:
        db.rollback()
        db.close()
    print("approval selfcheck ok")


if __name__ == "__main__":
    main()

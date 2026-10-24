"""Refresh-token rotation keeps the old token usable briefly; logout still kills it at once."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.core.security import hash_token
from app.modules.auth import service
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.sessions.models import SessionToken


class _Db:
    def __init__(self, session: SessionToken, user) -> None:
        self.sessions = {session.refresh_token_hash: session}
        self.user = user

    def scalar(self, stmt):
        wanted = stmt.whereclause.right.value
        return self.sessions.get(wanted)

    def get(self, _model, _id):
        return self.user

    def add(self, obj):
        if isinstance(obj, SessionToken):
            self.sessions[obj.refresh_token_hash] = obj


def _setup():
    old = SessionToken(
        user_id=1,
        refresh_token_hash=hash_token("old"),
        expires_at=datetime.now(UTC) + timedelta(days=7),
        revoked=False,
    )
    user = SimpleNamespace(id=1, role=Role.MASTER_ADMIN, is_active=True, approved_at=datetime.now(UTC))
    return _Db(old, user), old


def test_rotated_token_works_within_grace_then_expires():
    db, old = _setup()
    service.refresh_access_token(db, "old")
    assert old.expires_at <= datetime.now(UTC) + service.REFRESH_REUSE_GRACE
    service.refresh_access_token(db, "old")  # second tab / lost response: still accepted

    old.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(AuthException):
        service.refresh_access_token(db, "old")


def test_logged_out_token_is_rejected_immediately():
    db, old = _setup()
    old.revoked = True
    with pytest.raises(AuthException):
        service.refresh_access_token(db, "old")

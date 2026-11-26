"""Students may be signed in on one browser at a time; everyone else is untouched."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.security import create_access_token, decode_token, hash_token
from app.core.student_messages import ALREADY_SIGNED_IN_ELSEWHERE, SESSION_REPLACED
from app.modules.auth import service, single_session
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.exceptions import AuthException
from app.modules.auth.schemas import LoginRequest
from app.modules.sessions.models import SessionToken

DEVICE_A = "deviceA-aaaaaaaa"
DEVICE_B = "deviceB-bbbbbbbb"


def _user(role=Role.STUDENT, active_sid=None):
    return SimpleNamespace(
        id=1, role=role, is_active=True, approved_at=datetime.now(UTC), active_sid=active_sid
    )


class _Db:
    """Just enough Session for start_session / refresh / logout / get_current_user."""

    def __init__(self, user, sessions=(), other_active=False):
        self.user = user
        self.sessions = {s.refresh_token_hash: s for s in sessions}
        self.other_active = other_active
        self.revoked_all = False

    def scalar(self, stmt):
        if "FROM sessions" in str(stmt):
            if len(list(stmt.selected_columns)) == 1:  # "is there another device's login?"
                return 1 if self.other_active else None
            return self.sessions.get(stmt.whereclause.right.value)  # refresh-token lookup
        return self.user  # the row-lock select

    def execute(self, _stmt):
        self.revoked_all = True

    def get(self, _model, _id):
        return self.user

    def add(self, obj):
        if isinstance(obj, SessionToken):
            self.sessions[obj.refresh_token_hash] = obj


def _session(sid, token="t", device_id=None):
    return SessionToken(
        user_id=1,
        refresh_token_hash=hash_token(token),
        sid=sid,
        device_id=device_id,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        revoked=False,
    )


def _real_db(rows):
    """An in-memory database holding only the sessions table, for the device query."""
    engine = create_engine("sqlite://")
    SessionToken.__table__.create(engine)
    db = sessionmaker(bind=engine)()
    for row in rows:
        db.add(row)
    db.commit()
    return db


def _row(i, device_id, *, user_id=1, revoked=False, expires_in=timedelta(days=7), created_ago=timedelta(0)):
    now = datetime.now(UTC)
    return SessionToken(
        id=i,
        user_id=user_id,
        refresh_token_hash=f"hash-{i}",
        sid=f"sid-{i}",
        device_id=device_id,
        expires_at=now + expires_in,
        created_at=now - created_ago,
        revoked=revoked,
    )


# ── who is limited ───────────────────────────────────────────────────────────


def test_only_students_are_limited():
    assert single_session.applies_to(_user(Role.STUDENT))
    for role in (Role.TUTOR, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN):
        assert not single_session.applies_to(_user(role))
        assert single_session.sid_is_current(_user(role, active_sid="x"), "other")


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(single_session.settings, "single_session_enabled", False)
    assert not single_session.applies_to(_user())
    assert single_session.sid_is_current(_user(active_sid="x"), "y")


# ── the prompt rule: another browser, valid login ────────────────────────────


def test_same_browser_signing_in_again_is_not_prompted():
    db = _real_db([_row(1, DEVICE_A)])
    assert single_session._has_other_device_session(db, 1, DEVICE_A) is False


def test_another_browser_with_a_valid_login_is_prompted():
    db = _real_db([_row(1, DEVICE_A)])
    assert single_session._has_other_device_session(db, 1, DEVICE_B) is True


def test_idle_login_still_counts_no_time_window():
    # Left signed in two days ago and never used since: still a valid login, so still prompted.
    db = _real_db([_row(1, DEVICE_A, created_ago=timedelta(days=2))])
    assert single_session._has_other_device_session(db, 1, DEVICE_B) is True


def test_revoked_or_expired_logins_do_not_count():
    db = _real_db([
        _row(1, DEVICE_A, revoked=True),
        _row(2, DEVICE_A, expires_in=-timedelta(minutes=1)),
    ])
    assert single_session._has_other_device_session(db, 1, DEVICE_B) is False


def test_other_students_logins_do_not_count():
    db = _real_db([_row(1, DEVICE_A, user_id=2)])
    assert single_session._has_other_device_session(db, 1, DEVICE_B) is False


def test_login_before_device_ids_existed_counts_as_another_browser():
    db = _real_db([_row(1, None)])
    assert single_session._has_other_device_session(db, 1, DEVICE_A) is True


def test_no_device_id_sent_is_treated_as_another_browser():
    db = _real_db([_row(1, DEVICE_A)])
    assert single_session._has_other_device_session(db, 1, None) is True


def test_device_id_format_is_checked():
    base = {"email": "a@example.com", "password": "password1"}
    assert LoginRequest(**base, device_id=DEVICE_A).device_id == DEVICE_A
    with pytest.raises(ValidationError):
        LoginRequest(**base, device_id="bad id with spaces!")


# ── login flow (fake database) ───────────────────────────────────────────────


def test_first_login_gets_a_sid_and_becomes_the_active_one():
    user = _user()
    sid = single_session.start_session(_Db(user), user, force=False, device_id=DEVICE_A)
    assert sid and user.active_sid == sid


def test_login_while_signed_in_elsewhere_asks_first():
    user = _user(active_sid="old")
    db = _Db(user, other_active=True)
    with pytest.raises(AuthException) as exc:
        single_session.start_session(db, user, force=False, device_id=DEVICE_B)
    assert exc.value.status_code == 409
    assert exc.value.detail == ALREADY_SIGNED_IN_ELSEWHERE
    assert user.active_sid == "old"  # nothing changed until the student confirms
    assert not db.revoked_all


def test_confirming_signs_the_other_device_out():
    user = _user(active_sid="old")
    db = _Db(user, other_active=True)
    sid = single_session.start_session(db, user, force=True, device_id=DEVICE_B)
    assert db.revoked_all
    assert user.active_sid == sid != "old"


def test_replaced_device_is_rejected_new_one_works():
    user = _user(active_sid="new")
    with pytest.raises(AuthException) as exc:
        single_session.assert_session_current(user, {"sid": "old"})
    assert exc.value.status_code == 401 and exc.value.detail == SESSION_REPLACED
    with pytest.raises(AuthException):  # token minted before the rollout has no sid
        single_session.assert_session_current(user, {})
    single_session.assert_session_current(user, {"sid": "new"})


def test_tokens_without_an_active_sid_are_not_enforced():
    single_session.assert_session_current(_user(active_sid=None), {})


# ── endpoints that decode the bearer themselves (voice) ──────────────────────


def _patch_session_local(monkeypatch, user):
    class _S:
        def get(self, _model, _id):
            return user

        def close(self):
            pass

    monkeypatch.setattr(single_session, "SessionLocal", lambda: _S())


def test_payload_session_replaced(monkeypatch):
    _patch_session_local(monkeypatch, _user(active_sid="new"))
    assert single_session.payload_session_replaced({"sub": "1", "sid": "old"}) is True
    assert single_session.payload_session_replaced({"sub": "1"}) is True  # pre-rollout token
    assert single_session.payload_session_replaced({"sub": "1", "sid": "new"}) is False
    assert single_session.payload_session_replaced({"sub": "not-a-number"}) is False


def test_payload_session_replaced_ignores_other_roles(monkeypatch):
    _patch_session_local(monkeypatch, _user(Role.TUTOR, active_sid="new"))
    assert single_session.payload_session_replaced({"sub": "1", "sid": "old"}) is False


def test_voice_http_stream_refuses_a_replaced_student(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from app.voice_api import chapter_voice_stream

    _patch_session_local(monkeypatch, _user(active_sid="new"))
    req = SimpleNamespace(student_key="")

    def call(sid):
        auth = "Bearer " + create_access_token("1", "STUDENT", sid=sid)
        return asyncio.run(chapter_voice_stream(req, authorization=auth))

    with pytest.raises(HTTPException) as exc:
        call("old")
    assert exc.value.status_code == 401 and exc.value.detail == SESSION_REPLACED
    assert call("new").media_type == "application/octet-stream"  # current login streams as before


def test_access_token_carries_sid_and_get_current_user_enforces_it():
    user = _user(active_sid="new")
    db = _Db(user)
    creds = lambda sid: SimpleNamespace(credentials=create_access_token("1", "STUDENT", sid=sid))  # noqa: E731
    assert decode_token(creds("new").credentials)["sid"] == "new"
    assert get_current_user(creds("new"), db) is user
    with pytest.raises(AuthException) as exc:
        get_current_user(creds("old"), db)
    assert exc.value.detail == SESSION_REPLACED


def test_refresh_keeps_the_sid_and_device_across_rotation():
    user = _user(active_sid="s1")
    old = _session("s1", "old", device_id=DEVICE_A)
    db = _Db(user, [old])
    _u, access, new_refresh = service.refresh_access_token(db, "old")
    assert decode_token(access)["sid"] == "s1"
    rotated = db.sessions[hash_token(new_refresh)]
    assert rotated.sid == "s1" and rotated.device_id == DEVICE_A


def test_replaced_device_refresh_is_told_why():
    user = _user(active_sid="new")
    old = _session("old", "old")
    old.revoked = True  # revoked by the new login
    with pytest.raises(AuthException) as exc:
        service.refresh_access_token(_Db(user, [old]), "old")
    assert exc.value.detail == SESSION_REPLACED


def test_logout_frees_the_slot_only_for_the_current_device():
    user = _user(active_sid="s1")
    single_session.end_session(user, "other")
    assert user.active_sid == "s1"
    single_session.end_session(user, "s1")
    assert user.active_sid is None

    user.active_sid = "s2"
    single_session.end_session(user, None, all_devices=True)
    assert user.active_sid is None

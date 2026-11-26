import pytest
from fastapi import HTTPException

from app.core.chat_rate_limit import check_chat_rate_limit, reset_chat_rate_limit


@pytest.fixture(autouse=True)
def _reset():
    reset_chat_rate_limit()
    yield
    reset_chat_rate_limit()


def test_allows_up_to_limit_then_429(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MIN", "3")
    for _ in range(3):
        check_chat_rate_limit(1)
    with pytest.raises(HTTPException) as exc:
        check_chat_rate_limit(1)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_limit_is_per_user(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MIN", "1")
    check_chat_rate_limit(1)
    check_chat_rate_limit(2)


def test_zero_disables(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MIN", "0")
    for _ in range(50):
        check_chat_rate_limit(1)

"""Deterministic temp passwords from full name + login user id (username)."""

from __future__ import annotations

import re

# LoginRequest.password requires min_length=8
_MIN_LEN = 8


def make_credential_password(full_name: str, username: str) -> str:
    """Build a shareable temp password from the user's name and user id.

    Example: "Anita Verma" + "anita.verma" → "Av@averma" (always ≥ 8 chars for login).
    """
    parts = [p for p in re.split(r"\W+", (full_name or "").strip()) if p]
    if not parts:
        parts = [username or "user"]
    first = parts[0]
    second = parts[1] if len(parts) > 1 else first
    a = first[0].upper()
    b = (second[0] if second else "x").lower()
    uid = "".join(c for c in (username or "") if c.isalnum()).lower() or "user"
    # 6-char suffix → "Xy@xxxxxx" is always 9 chars (≥ login min 8)
    suffix = (uid[-6:] if len(uid) >= 6 else (uid + "000000")[:6])
    password = f"{a}{b}@{suffix}"
    if len(password) < _MIN_LEN:
        password = (password + "0" * _MIN_LEN)[:_MIN_LEN]
    return password


def _self_check() -> None:
    pw = make_credential_password("Anita Verma", "anita.verma")
    assert pw == "Av@averma", pw
    assert len(pw) >= _MIN_LEN
    bob = make_credential_password("Bob", "bob01")
    assert bob == "Bb@bob010", bob
    assert len(bob) >= _MIN_LEN


if __name__ == "__main__":
    _self_check()
    print("ok", make_credential_password("Anita Verma", "anita.verma"))

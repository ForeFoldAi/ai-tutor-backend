"""LIA auth dependencies."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def require_lia_internal_token(x_lia_token: str | None = Header(default=None, alias="X-LIA-Token")) -> None:
    expected = os.environ.get("LIA_INTERNAL_TOKEN", "")
    if not expected:
        # ponytail: dev mode — skip if unset; production must set LIA_INTERNAL_TOKEN
        return
    if not x_lia_token or x_lia_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid LIA token")

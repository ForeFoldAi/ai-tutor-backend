from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Header, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_token
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.users.models import User

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if not credentials:
        raise AuthException("Not authenticated.", status.HTTP_401_UNAUTHORIZED)
    try:
        payload = decode_token(credentials.credentials)
    except ValueError as exc:
        raise AuthException("Invalid access token.", status.HTTP_401_UNAUTHORIZED) from exc
    if payload.get("type") != "access":
        raise AuthException("Invalid access token.", status.HTTP_401_UNAUTHORIZED)
    user_id = payload.get("sub")
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_401_UNAUTHORIZED)
    return user


def get_current_user_bearer_or_query(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
    access_token: str | None = Query(None, description="Optional JWT for <img src> loads"),
) -> User:
    """Same as get_current_user but allows passing the access JWT as *access_token* (e.g. image tags)."""
    raw: str | None = None
    if credentials and credentials.credentials:
        raw = credentials.credentials
    elif access_token:
        raw = access_token
    if not raw:
        raise AuthException("Not authenticated.", status.HTTP_401_UNAUTHORIZED)
    try:
        payload = decode_token(raw)
    except ValueError as exc:
        raise AuthException("Invalid access token.", status.HTTP_401_UNAUTHORIZED) from exc
    if payload.get("type") != "access":
        raise AuthException("Invalid access token.", status.HTTP_401_UNAUTHORIZED)
    user_id = payload.get("sub")
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_401_UNAUTHORIZED)
    return user


def require_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not current_user.is_active:
        raise AuthException("Inactive account.", status.HTTP_403_FORBIDDEN)
    return current_user


def require_roles(*roles: Role) -> Callable:
    allowed = set(roles)

    def _dep(current_user: Annotated[User, Depends(require_active_user)]) -> User:
        if current_user.role not in allowed:
            raise AuthException("Insufficient permissions.", status.HTTP_403_FORBIDDEN)
        return current_user

    return _dep


def get_request_context(
    user_agent: str | None = Header(default=None),
    x_forwarded_for: str | None = Header(default=None),
) -> dict[str, str | None]:
    return {"user_agent": user_agent, "ip_address": x_forwarded_for}

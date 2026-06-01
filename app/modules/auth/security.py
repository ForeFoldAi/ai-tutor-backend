import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import generate_refresh_token, hash_token
from app.modules.sessions.models import SessionToken

settings = get_settings()


def create_refresh_session(
    db: Session,
    user_id: uuid.UUID,
    user_agent: str | None,
    ip_address: str | None,
) -> tuple[str, SessionToken]:
    refresh_token = generate_refresh_token()
    refresh_hash = hash_token(refresh_token)
    expires_at = datetime.now(UTC) + timedelta(days=settings.refresh_token_exp_days)
    session = SessionToken(
        user_id=user_id,
        refresh_token_hash=refresh_hash,
        user_agent=user_agent,
        ip_address=ip_address,
        expires_at=expires_at,
        revoked=False,
    )
    db.add(session)
    return refresh_token, session

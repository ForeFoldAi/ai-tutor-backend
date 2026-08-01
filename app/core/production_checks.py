"""
Production startup guards — fail fast on unsafe defaults.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_WEAK_JWT = frozenset(
    {
        "",
        "change-me-in-production",
        "replace-with-strong-secret",
        "dev-secret",
        "secret",
    }
)


def _is_production() -> bool:
    env = os.environ.get("ENVIRONMENT", os.environ.get("APP_ENV", "development")).strip().lower()
    return env in ("production", "prod", "live")


def validate_production_settings() -> None:
    """Raise on critical misconfiguration when ENVIRONMENT=production."""
    if not _is_production():
        logger.info("ENVIRONMENT=%s — production checks skipped", os.environ.get("ENVIRONMENT", "development"))
        return

    from app.config import LLM_API_KEY, MISTRAL_API_KEY
    from app.core.config import get_settings

    settings = get_settings()
    errors: list[str] = []

    secret = (settings.jwt_secret_key or "").strip()
    if secret in _WEAK_JWT or len(secret) < 32:
        errors.append("JWT_SECRET_KEY must be a strong random string (32+ characters)")

    if not (LLM_API_KEY or MISTRAL_API_KEY):
        errors.append("LLM_API_KEY or MISTRAL_API_KEY must be set")

    raw_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if raw_origins == "*" or not raw_origins:
        errors.append(
            "ALLOWED_ORIGINS must list explicit frontend URLs in production "
            "(wildcard * is not allowed with credentials)"
        )

    if settings.smtp_enabled and not settings.smtp_password.strip():
        errors.append("SMTP_PASSWORD is required when SMTP_ENABLED=true")

    if errors:
        msg = "Production configuration errors:\n- " + "\n- ".join(errors)
        raise RuntimeError(msg)

    logger.info("Production configuration checks passed")

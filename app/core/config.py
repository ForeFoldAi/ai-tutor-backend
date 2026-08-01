import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Celery workers don't import app.config — load .env here so DATABASE_URL is set.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _normalize_database_url(url: str) -> str:
    """Heroku-style `postgres://` and driverless `postgresql://` need an explicit psycopg2 dialect."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if scheme == "postgres" or scheme == "postgresql":
        return f"postgresql+psycopg2://{rest}"
    return url


class Settings(BaseModel):
    # default_factory so normalization runs (validators do not run on plain field defaults).
    database_url: str = Field(
        default_factory=lambda: _normalize_database_url(
            os.environ.get(
                "DATABASE_URL",
                "postgresql+psycopg2://postgres:postgres@localhost:5432/ai_tutor",
            )
        )
    )

    @field_validator("database_url", mode="after")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        return _normalize_database_url(v)
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    jwt_secret_key: str = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = os.environ.get("JWT_ALGORITHM", "HS256")
    access_token_exp_minutes: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    refresh_token_exp_days: int = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    app_name: str = os.environ.get("APP_NAME", "AI Tutor")
    api_prefix: str = os.environ.get("API_PREFIX", "")

    smtp_enabled: bool = os.environ.get("SMTP_ENABLED", "false").lower() in ("1", "true", "yes")
    smtp_host: str = os.environ.get("SMTP_HOST", "localhost")
    smtp_port: int = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user: str = os.environ.get("SMTP_USER", "")
    smtp_password: str = os.environ.get("SMTP_PASSWORD", "")
    smtp_from: str = os.environ.get("SMTP_FROM", "noreply@example.com")
    smtp_tls: bool = os.environ.get("SMTP_TLS", "true").lower() in ("1", "true", "yes")
    # ponytail: false for local MITM / self-signed chains; keep true in production
    smtp_validate_certs: bool = os.environ.get("SMTP_VALIDATE_CERTS", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    password_otp_expire_minutes: int = int(os.environ.get("PASSWORD_OTP_EXPIRE_MINUTES", "10"))
    password_otp_length: int = int(os.environ.get("PASSWORD_OTP_LENGTH", "6"))
    password_otp_max_attempts: int = int(os.environ.get("PASSWORD_OTP_MAX_ATTEMPTS", "5"))

    mail_reminders_enabled: bool = os.environ.get("MAIL_REMINDERS_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    # Sessions 08:00 IST, assignments 09:00 IST by default
    mail_session_reminder_hour: int = int(os.environ.get("MAIL_SESSION_REMINDER_HOUR", "8"))
    mail_assignment_reminder_hour: int = int(os.environ.get("MAIL_ASSIGNMENT_REMINDER_HOUR", "9"))
    mail_reminder_minute: int = int(os.environ.get("MAIL_REMINDER_MINUTE", "0"))
    mail_reminder_tz: str = os.environ.get("MAIL_REMINDER_TZ", "Asia/Kolkata")
    # Gap between individual reminder emails (avoid bulk blast)
    mail_reminder_send_gap_seconds: float = float(os.environ.get("MAIL_REMINDER_SEND_GAP_SECONDS", "1"))

    # Local convenience: spawn Celery worker + beat when uvicorn starts
    celery_autostart: bool = os.environ.get("CELERY_AUTOSTART", "false").lower() in (
        "1",
        "true",
        "yes",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

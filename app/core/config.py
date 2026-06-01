import os
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator


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

    app_name: str = os.environ.get("APP_NAME", "AI Tutor Backend")
    api_prefix: str = os.environ.get("API_PREFIX", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()

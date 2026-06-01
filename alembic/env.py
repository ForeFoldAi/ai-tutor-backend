"""Alembic environment. Load `.env` before any `app` imports — `app.core.database` reads settings at import time."""
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent of this `alembic/` directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.core.database import Base
from app.modules.organizations.models import Organization  # noqa: F401
from app.modules.schools.models import School  # noqa: F401
from app.modules.sessions.models import SessionToken  # noqa: F401
from app.modules.users.models import User  # noqa: F401
from app.modules.catalog.models import BoardDefinition, SyllabusSubject, TextbookUpload  # noqa: F401

config = context.config
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

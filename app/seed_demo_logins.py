"""
Ensure demo logins exist for every role (STUDENT, TUTOR, SCHOOL_ADMIN, ORG_ADMIN, MASTER_ADMIN).

Run from the backend root after schema migrations:

    .venv/bin/python -m app.seed_demo_logins

Uses VITE_TEST_* (or TEST_*) username/password variables from `.env`; emails are
`<username>@example.com`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.modules.auth.bootstrap import seed_test_users_if_missing  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    seed_test_users_if_missing()
    logger.info(
        "Demo logins synced. Sign in with usernames from .env "
        "(or full emails like student@example.com); password from matching VITE_TEST_*_PASSWORD."
    )


if __name__ == "__main__":
    main()

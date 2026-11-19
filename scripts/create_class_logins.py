#!/usr/bin/env python3
"""One approved student login per class for a board (classes 5-10): ssc5..ssc10 or cbse5..cbse10.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/create_class_logins.py                      # State Board sample only
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/create_class_logins.py --create             # State Board, write to DATABASE_URL
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/create_class_logins.py --board cbse --create  # CBSE

Uses the public signup path, then approves the account. Existing usernames are skipped.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_ROOT / ".env")

from app.core.database import SessionLocal  # noqa: E402
from app.modules.auth.signup.schemas import StudentSignupRequest  # noqa: E402
from app.modules.auth.signup.service import signup_student  # noqa: E402
from app.modules.auth.signup.validators import find_user_by_username  # noqa: E402

# username prefix -> curriculum value accepted by signup
BOARDS = {"ssc": "State Board", "cbse": "CBSE"}
CLASSES = range(5, 11)


def accounts(prefix: str) -> list[dict]:
    return [
        {
            "full_name": f"{prefix.upper()} Class {n}",
            "user_id": f"{prefix}{n}",
            "password": "12345678",
            "grade": str(n),
            "curricula": [BOARDS[prefix]],
            # 'Science' covers EVS / General / Physical / Biological Science / Environmental Education.
            "favorite_subjects": ["Mathematics", "English", "Science"] + (["Social"] if n >= 6 else []),
        }
        for n in CLASSES
    ]


def main() -> int:
    create = "--create" in sys.argv
    prefix = sys.argv[sys.argv.index("--board") + 1].lower() if "--board" in sys.argv else "ssc"
    if prefix not in BOARDS:
        print(f"--board must be one of: {', '.join(BOARDS)}")
        return 2
    board = BOARDS[prefix]
    db = SessionLocal()
    try:
        print(f"{'username':12} {'password':12} {'class':6} board")
        for a in accounts(prefix):
            exists = find_user_by_username(db, a["user_id"]) is not None
            status = "exists, skipped" if exists else ("created" if create else "sample")
            if create and not exists:
                user = signup_student(db, StudentSignupRequest(**a))
                user.approved_at = datetime.now(UTC)
                user.is_verified = True
            print(f"{a['user_id']:12} {a['password']:12} {a['grade']:6} {board}  [{status}]")
        if create:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

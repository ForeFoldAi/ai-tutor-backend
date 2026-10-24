#!/usr/bin/env python3
"""One approved student login per class for a board (State Board, classes 5-10).

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/create_class_logins.py            # print sample only
  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/create_class_logins.py --create   # write to DATABASE_URL

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

BOARD = "State Board"
CLASSES = range(5, 11)


def accounts() -> list[dict]:
    return [
        {
            "full_name": f"SSC Class {n}",
            "user_id": f"ssc{n}",
            "password": "12345678",
            "grade": str(n),
            "curricula": [BOARD],
            # 'Science' covers EVS / General / Physical / Biological Science / Environmental Education.
            "favorite_subjects": ["Mathematics", "English", "Science"] + (["Social"] if n >= 6 else []),
        }
        for n in CLASSES
    ]


def main() -> int:
    create = "--create" in sys.argv
    db = SessionLocal()
    try:
        print(f"{'username':12} {'password':12} {'class':6} board")
        for a in accounts():
            exists = find_user_by_username(db, a["user_id"]) is not None
            status = "exists, skipped" if exists else ("created" if create else "sample")
            if create and not exists:
                user = signup_student(db, StudentSignupRequest(**a))
                user.approved_at = datetime.now(UTC)
                user.is_verified = True
            print(f"{a['user_id']:12} {a['password']:12} {a['grade']:6} {BOARD}  [{status}]")
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

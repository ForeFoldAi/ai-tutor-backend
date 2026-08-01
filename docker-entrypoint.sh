#!/bin/sh
set -e

echo "Waiting for PostgreSQL..."
python - <<'PY'
import os
import sys
import time
from urllib.parse import urlparse

import psycopg2

raw = os.environ.get("DATABASE_URL", "")
raw = raw.replace("postgresql+psycopg2://", "postgresql://").replace("postgres://", "postgresql://")
parsed = urlparse(raw)
if not parsed.hostname:
    sys.exit("DATABASE_URL is not set or invalid")

for attempt in range(60):
    try:
        psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            dbname=(parsed.path or "/ai_tutor").lstrip("/"),
        ).close()
        break
    except psycopg2.OperationalError:
        time.sleep(1)
else:
    sys.exit("PostgreSQL did not become ready in time")
PY

mkdir -p /app/models/Layout/YOLO /app/models/MFD/YOLO

# ponytail: skip by default — Railway/prod DB is already migrated; set RUN_MIGRATIONS=1 to enable
if [ "${RUN_MIGRATIONS:-0}" = "1" ]; then
  echo "Running database migrations..."
  alembic upgrade head
else
  echo "Skipping database migrations (RUN_MIGRATIONS!=1)"
fi

echo "Starting API..."
# Railway (and most PaaS) inject PORT; local/Docker Compose default to 8000.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"

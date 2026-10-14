#!/bin/sh
set -e

mkdir -p /app/models/Layout/YOLO /app/models/MFD/YOLO

echo "Starting API (Nuitka-compiled app module)..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"

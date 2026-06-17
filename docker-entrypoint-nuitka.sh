#!/bin/sh
set -e

mkdir -p /app/models/Layout/YOLO /app/models/MFD/YOLO

echo "Starting API (Nuitka-compiled)..."
exec /app/nuitka.dist/nuitka_launcher.bin "$@"

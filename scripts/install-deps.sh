#!/usr/bin/env bash
# ponytail: two-step install avoids pip resolution-too-deep on the main lockfile
set -euo pipefail
cd "$(dirname "$0")/.."
PIP="${PIP:-pip}"
echo "==> Step 1/2: core requirements (may take several minutes)..."
"$PIP" install -r requirements.txt
echo "==> Step 2/2: lesson planner..."
"$PIP" install -r requirements-lesson-planner.txt
echo "Done. Optional: $PIP install -r requirements-lesson-planner-optional.txt"

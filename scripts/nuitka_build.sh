#!/bin/sh
set -e

# Compile only the application package. Third-party deps stay as pip wheels at
# runtime (avoids aarch64 linker overflow on huge standalone binaries).
python -m nuitka \
    --module \
    --assume-yes-for-downloads \
    --output-dir=/build/nuitka-out \
    --include-package=app \
    --include-data-dir=configs=configs \
    --include-data-files=Ultralytics/settings.yaml=Ultralytics/settings.yaml \
    app

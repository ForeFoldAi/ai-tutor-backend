#!/bin/sh
set -e

# Heavy ML / native extensions are kept as runtime pip packages (see Dockerfile.nuitka).
NOFOLLOW="
torch
torchvision
transformers
sentence_transformers
chromadb
paddle
paddleocr
cv2
ultralytics
doclayout_yolo
onnxruntime
sklearn
scipy
PIL
fitz
pymupdf
edge_tts
soundfile
redis
psycopg2
numpy
tokenizers
safetensors
huggingface_hub
langchain
langchain_community
langchain_core
langgraph
"

NOFOLLOW_ARGS=""
for pkg in $NOFOLLOW; do
    NOFOLLOW_ARGS="$NOFOLLOW_ARGS --nofollow-import-to=$pkg"
done

python -m nuitka \
    --standalone \
    --assume-yes-for-downloads \
    --output-dir=/build/nuitka-out \
    --include-package=app \
    --include-data-dir=configs=configs \
    --include-data-files=Ultralytics/settings.yaml=Ultralytics/settings.yaml \
    --nofollow-import-to=alembic \
    --nofollow-import-to=alembic.testing \
    --nofollow-import-to=sqlalchemy.testing \
    $NOFOLLOW_ARGS \
    scripts/nuitka_launcher.py

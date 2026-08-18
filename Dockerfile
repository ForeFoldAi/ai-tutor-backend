FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDF_EXTRACTION_MODELS_DIR=/app/models \
    PDF_EXTRACTION_PIPELINE_ENABLED=true \
    OCR_ENABLED=true \
    USE_MULTIMODAL_IMAGE_RETRIEVAL=true \
    WARM_MULTIMODAL_ON_STARTUP=false \
    CELERY_AUTOSTART=false \
    ENABLE_VISUAL_INTENT_DETECTION=true

# Native libs for psycopg2, PyMuPDF, OpenCV, PaddleOCR, Tesseract, and torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq-dev \
    libsm6 \
    libxext6 \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswresample-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt requirements-ml-docker.txt requirements-lesson-planner.txt requirements-voice-protection.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-docker.txt \
    && pip install -r requirements-ml-docker.txt \
    && pip install -r requirements-lesson-planner.txt \
    && pip install -r requirements-voice-protection.txt

# Bake Silero VAD into the image so production cold-start does not need torch.hub network.
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', model='silero_vad', trust_repo=True, onnx=False)"

COPY . .
RUN chmod +x docker-entrypoint.sh

RUN mkdir -p uploads chroma_db models/Layout/YOLO models/MFD/YOLO

EXPOSE 8000

# CLIP / Edge TTS warm-up can take a few minutes on cold start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1'

ENTRYPOINT ["./docker-entrypoint.sh"]

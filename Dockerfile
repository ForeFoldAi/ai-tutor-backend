# Tesseract 5.5.3 (latest upstream, includes CVE-2026-73066/73067 fixes). Debian bookworm only
# ships 5.3.0, so build the official release tarball here; only the binary, library and
# tessdata (configs + English) are copied into the runtime image.
FROM python:3.11-slim-bookworm AS tesseract-build
ARG TESSERACT_VERSION=5.5.3
ARG TESSERACT_SHA256=9218e62793116d42a9f6d14cd9348518b27f382096eea3d0f2d1a24616bb5884
ARG TESSDATA_FAST_TAG=4.1.0
ARG ENG_TRAINEDDATA_SHA256=7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake ca-certificates curl g++ libleptonica-dev libtool make pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o tesseract.tar.gz "https://github.com/tesseract-ocr/tesseract/archive/refs/tags/${TESSERACT_VERSION}.tar.gz" \
    && echo "${TESSERACT_SHA256}  tesseract.tar.gz" | sha256sum -c - \
    && tar xzf tesseract.tar.gz \
    && cd "tesseract-${TESSERACT_VERSION}" \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local --disable-doc --disable-graphics --disable-openmp --without-curl --without-archive \
    && make -j"$(nproc)" \
    && make install \
    && curl -fsSL -o /usr/local/share/tessdata/eng.traineddata \
        "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/${TESSDATA_FAST_TAG}/eng.traineddata" \
    && echo "${ENG_TRAINEDDATA_SHA256}  /usr/local/share/tessdata/eng.traineddata" | sha256sum -c -

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
    ENABLE_VISUAL_INTENT_DETECTION=true \
    TESSDATA_PREFIX=/usr/local/share/tessdata

# Native libs for psycopg2, PyMuPDF, OpenCV, PaddleOCR, Tesseract (liblept5), and torch.
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
    liblept5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tesseract-build /usr/local/bin/tesseract /usr/local/bin/tesseract
COPY --from=tesseract-build /usr/local/lib/libtesseract.so* /usr/local/lib/
COPY --from=tesseract-build /usr/local/share/tessdata /usr/local/share/tessdata
RUN ldconfig \
    && tesseract --version 2>&1 | head -1 | grep -q "tesseract 5.5.3" \
    && tesseract --list-langs 2>&1 | grep -qx eng

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

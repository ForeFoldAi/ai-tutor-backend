# AI Tutor Backend

FastAPI backend for chapter-aware tutoring, textbook catalog/uploads, RAG chat, voice APIs, and textbook figure retrieval.

## Prerequisites

- **Python 3.11+** (3.14 works with the existing `.venv` in this repo)
- **PostgreSQL** (default DB: `ai_tutor`)
- **Redis** (optional but recommended for caching)
- **Mistral API key** for LLM answers (`MISTRAL_API_KEY` in `.env`)

## Setup

From the `ai-tutor-backend` directory:

### 1. Create and activate the virtual environment

```bash
cd ai-tutor-backend

python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Optional — ML PDF extraction (DocLayout-YOLO, OCR, tables, formulas):

```bash
.venv/bin/pip install -r requirements-ml.txt
```

Layout weights (`models/Layout/YOLO/doclayout_yolo_ft.pt`) are **not** in git. On first run the pipeline downloads from Hugging Face (`juliozhao/DocLayout-YOLO-DocStructBench`) or you can place the `.pt` file locally.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

- `MISTRAL_API_KEY`
- `DATABASE_URL`
- `JWT_SECRET_KEY`

### 4. Run database migrations

```bash
.venv/bin/alembic upgrade head
```

## Run

### Development server (with auto-reload)

```bash
cd ai-tutor-backend
source .venv/bin/activate

.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API base URL: `http://127.0.0.1:8000`

Health check:

```bash
curl http://127.0.0.1:8000/health/ai-models
```

Interactive docs: `http://127.0.0.1:8000/docs`

### Production-style (no reload)

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

Run the **full backend** (tutor chat, voice, Redis cache, multimodal CLIP image search, ML PDF extraction, PaddleOCR, Tesseract OCR) with Docker Compose.

### 1. Configure environment

```bash
cd ai-tutor-backend
cp .env.example .env
```

Edit `.env` and set at minimum:

- `MISTRAL_API_KEY` — LLM answers (required)
- `JWT_SECRET_KEY` — auth tokens (required)
- `HF_TOKEN` — download CLIP + DocLayout-YOLO weights (strongly recommended)

Compose sets `DATABASE_URL` and `REDIS_URL` for the `db` and `redis` services. Feature flags for multimodal images, PDF ML pipeline, and OCR are enabled in `docker-compose.yml`.

### 2. Build and start

```bash
docker compose up --build -d
```

The image installs `requirements-docker.txt` and `requirements-ml-docker.txt` (excludes `llama_cpp_python` and desktop-only packages; chat uses Mistral API). First build may take **15–30+ minutes** (PyTorch, Paddle, DocLayout-YOLO, etc.).

API: `http://127.0.0.1:8000`  
Docs: `http://127.0.0.1:8000/docs`

Health check:

```bash
curl http://127.0.0.1:8000/health/ai-models
```

Logs:

```bash
docker compose logs -f api
```

Stop:

```bash
docker compose down
```

### What is included

| Feature | Docker default |
|---------|----------------|
| FastAPI tutor + auth + catalog | Yes |
| Chapter-aware RAG (ChromaDB) | Yes |
| Voice WebSocket + Edge TTS | Yes |
| Redis answer cache | Yes |
| Multimodal CLIP image retrieval | Yes (`HF_TOKEN` needed) |
| ML PDF extraction (DocLayout-YOLO, formulas) | Yes (layout + OCR; formula LaTeX on ARM Docker skips UniMERNet) |
| PaddleOCR + Tesseract OCR | Yes |
| Table VLM (`struct-eqtable`) | No — requires NVIDIA GPU |

### Persistent volumes

| Volume | Purpose |
|--------|---------|
| `pg_data` | PostgreSQL |
| `uploads_data` | Textbook PDFs and extracted figures |
| `chroma_data` | Vector embeddings |
| `models_data` | Downloaded layout / formula weights |
| `hf_cache` | Hugging Face model cache (CLIP, etc.) |

Optional: copy a local YOLO weight into the models volume:

```bash
docker cp models/Layout/YOLO/doclayout_yolo_ft.pt \
  $(docker compose ps -q api):/app/models/Layout/YOLO/doclayout_yolo_ft.pt
```

Otherwise the layout model downloads from Hugging Face on first PDF extraction.

### Build image only (without Compose)

```bash
docker build -t ai-tutor-backend:latest .
```

### Docker Hub (publish for your team)

**Workflow: build locally → test → push.** Full steps: **`deploy/PUBLISH.md`**.

**You (build & push once):**

```bash
cd ai-tutor-backend

# 1. Log in to Docker Hub
docker login

# 2. Build (use your Docker Hub username)
export DOCKERHUB_USER=yourdockerhubuser
docker build -t ${DOCKERHUB_USER}/ai-tutor-backend:latest .

# 3. Push
docker push ${DOCKERHUB_USER}/ai-tutor-backend:latest
```

**Team on mixed Mac + Linux?** Build for both architectures:

```bash
docker buildx create --use   # once
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ${DOCKERHUB_USER}/ai-tutor-backend:latest \
  --push .
```

Share with the team:
- Image name: `yourdockerhubuser/ai-tutor-backend:latest`
- Folder `deploy/` from this repo (`docker-compose.yml` + `TEAM_SETUP.md`)

**Your team (pull & run with their `.env`):**

```bash
cd deploy

API_IMAGE=yourdockerhubuser/ai-tutor-backend:latest \
ENV_FILE=/absolute/path/to/their.env \
docker compose up -d
```

They only need `MISTRAL_API_KEY`, `JWT_SECRET_KEY`, and `HF_TOKEN` in `.env`.  
Postgres and Redis are started automatically; no build step.

See **`deploy/TEAM_SETUP.md`** for the full team guide.

### Portable image (build once, run on another machine with your `.env`)

Use this when you want to **build the image on your dev machine**, move it to a server, and pass a **custom `.env` path** at runtime.

#### Step 1 — Build the image (on your machine)

```bash
cd ai-tutor-backend
docker build -t ai-tutor-backend:latest .
```

#### Step 2 — Export the image (optional, if not using a registry)

```bash
docker save ai-tutor-backend:latest | gzip > ai-tutor-backend.tar.gz
```

Copy `ai-tutor-backend.tar.gz` and `docker-compose.image.yml` to the other machine.

#### Step 3 — Load the image (on the other machine)

```bash
docker load < ai-tutor-backend.tar.gz
```

Or push/pull via Docker Hub / GHCR:

```bash
docker tag ai-tutor-backend:latest youruser/ai-tutor-backend:latest
docker push youruser/ai-tutor-backend:latest
# on other machine:
docker pull youruser/ai-tutor-backend:latest
export API_IMAGE=youruser/ai-tutor-backend:latest
```

#### Step 4 — Run with your `.env` file path

**Recommended** — API + Postgres + Redis (Compose overrides `DATABASE_URL` / `REDIS_URL` to internal services):

```bash
ENV_FILE=/absolute/path/to/.env docker compose -f docker-compose.image.yml up -d
```

Your `.env` must include at least:

```env
MISTRAL_API_KEY=...
JWT_SECRET_KEY=...
HF_TOKEN=...
```

`DATABASE_URL` and `REDIS_URL` in `.env` are ignored when using `docker-compose.image.yml` (Compose sets them to `db` and `redis`).

**API container only** — if Postgres and Redis already run elsewhere:

```bash
docker run -d --name ai-tutor-api \
  -p 8000:8000 \
  --env-file /absolute/path/to/.env \
  -e DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:5432/ai_tutor \
  -e REDIS_URL=redis://HOST:6379/0 \
  -v ai_tutor_uploads:/app/uploads \
  -v ai_tutor_chroma:/app/chroma_db \
  -v ai_tutor_models:/app/models \
  -v ai_tutor_hf_cache:/root/.cache/huggingface \
  ai-tutor-backend:latest
```

Replace `HOST`, `USER`, and `PASS` with your database and Redis endpoints.

Check logs:

```bash
docker compose -f docker-compose.image.yml logs -f api
# or for docker run:
docker logs -f ai-tutor-api
```

### Run a built image manually

You must provide Postgres and Redis (or use Compose for `db` / `redis` only):

```bash
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql+psycopg2://postgres:postgres@host.docker.internal:5432/ai_tutor \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -e MISTRAL_API_KEY=your-key \
  -e JWT_SECRET_KEY=your-secret \
  -e HF_TOKEN=your-hf-token \
  -v ai_tutor_uploads:/app/uploads \
  -v ai_tutor_chroma:/app/chroma_db \
  -v ai_tutor_models:/app/models \
  -v ai_tutor_hf_cache:/root/.cache/huggingface \
  ai-tutor-backend:latest
```

On Linux, replace `host.docker.internal` with your database host IP.

### Docker troubleshooting

- **First startup is slow** — migrations, CLIP warm-up, and Hugging Face downloads on first use.
- **CLIP unavailable** — set `HF_TOKEN` in `.env` and restart: `docker compose up -d --build api`.
- **PDF layout model** — stored in `models_data` volume; not committed to git.
- **Port already in use** — set `API_PORT=8001` before `docker compose up`.
- **Out of memory** — the full image needs **8 GB+ RAM** recommended for ML extraction.
- **`llama_cpp_python` build errors on Apple Silicon** — the Docker image no longer installs it; use Mistral API (`MISTRAL_API_KEY`) instead of local llama.cpp.

## Frontend connection

Point the frontend at this server (in `ai-tutor-frontend/.env`):

```env
VITE_API_URL=http://127.0.0.1:8000
VITE_VOICE_URL=http://127.0.0.1:8000
```

## Useful commands

All commands assume you are in `ai-tutor-backend` and use the project `.venv`.

```bash
# Activate venv (once per terminal session)
source .venv/bin/activate

# Run tests (install pytest first if needed)
.venv/bin/pip install pytest
.venv/bin/pytest tests/ -q

# Re-extract textbook images for an upload
PYTHONPATH=. .venv/bin/python scripts/reextract_textbook_images.py --upload-id <uuid>

# Test image extraction against a catalog upload
.venv/bin/python scripts/test_image_extraction.py

# Local ML extraction on a PDF file (no database)
.venv/bin/python scripts/test_pdf_extraction_local.py path/to/file.pdf
```

## Project layout

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI app entrypoint |
| `app/config.py` | Tutor / RAG / image retrieval settings |
| `app/modules/` | Auth, catalog, users, schools |
| `app/services/` | Chat, RAG, PDF extraction, image retrieval |
| `alembic/` | Database migrations |
| `uploads/` | Uploaded textbooks and extracted figures |
| `Dockerfile` | Container image for the API |
| `docker-compose.yml` | API + PostgreSQL + Redis stack (build from source) |
| `docker-compose.image.yml` | Same stack using a pre-built image + `ENV_FILE` |
| `.env` | Local secrets and configuration (not committed) |

## Troubleshooting

- **Database connection errors** — confirm PostgreSQL is running and `DATABASE_URL` matches your local setup.
- **CLIP / multimodal image search unavailable** — set `HF_TOKEN` in `.env` when `USE_MULTIMODAL_IMAGE_RETRIEVAL=true`.
- **CORS issues** — set `ALLOWED_ORIGINS` in `.env` to your frontend origin (comma-separated).

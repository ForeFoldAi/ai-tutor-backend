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
| `.env` | Local secrets and configuration (not committed) |

## Troubleshooting

- **Database connection errors** — confirm PostgreSQL is running and `DATABASE_URL` matches your local setup.
- **CLIP / multimodal image search unavailable** — set `HF_TOKEN` in `.env` when `USE_MULTIMODAL_IMAGE_RETRIEVAL=true`.
- **CORS issues** — set `ALLOWED_ORIGINS` in `.env` to your frontend origin (comma-separated).

# Team setup — AI Tutor backend (Docker Hub)

Run the backend from Docker Hub. **No build, no full repo clone required.**

**Image:** `manithmettu/ai-tutor-backend:latest`  
**Setup:** API + Redis in Docker; **Postgres is your shared cloud database** (from `.env`).

---

## What you need

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or Docker Engine (Linux)
2. Two files from the repo `deploy/` folder:
   - `docker-compose.external.yml`
   - (this guide)
3. A `.env` file with secrets (get values from your team lead)

---

## Step 1 — Install Docker

Install and start Docker Desktop. Verify:

```bash
docker --version
docker compose version
```

---

## Step 2 — Get the compose file

Copy `deploy/docker-compose.external.yml` to a folder on your machine, e.g.:

```bash
mkdir -p ~/ai-tutor-deploy
cd ~/ai-tutor-deploy
# copy docker-compose.external.yml here (from repo or shared drive)
```

---

## Step 3 — Create your `.env` file

Create `~/ai-tutor.env` (any path is fine):

```env
# Required — get from team lead
MISTRAL_API_KEY=your-mistral-api-key
JWT_SECRET_KEY=your-long-random-secret
HF_TOKEN=your-huggingface-token

# Required — shared cloud Postgres (team lead will provide host/user/password)
DATABASE_URL=postgres://postgres:PASSWORD@HOST:5434/tutor

# Optional
MISTRAL_MODEL=mistral-small-latest
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Notes:
- `DATABASE_URL` must point to the **shared cloud Postgres** (not `localhost`).
- `REDIS_URL` in `.env` is **ignored** — Redis runs in Docker automatically.
- Do **not** commit `.env` to git.

---

## Step 4 — Log in to Docker Hub (private repo only)

Skip if the image is public.

```bash
docker login
```

Use your Docker Hub username and password (or access token).

---

## Step 5 — Pull and start

```bash
cd ~/ai-tutor-deploy

export API_IMAGE=manithmettu/ai-tutor-backend:latest
export ENV_FILE=~/ai-tutor.env

docker compose -f docker-compose.external.yml pull
docker compose -f docker-compose.external.yml up -d
```

One-liner:

```bash
API_IMAGE=manithmettu/ai-tutor-backend:latest ENV_FILE=~/ai-tutor.env docker compose -f docker-compose.external.yml up -d
```

First start downloads the image (~12 GB) and may take several minutes.

---

## Step 6 — Verify it works

Wait 1–3 minutes, then:

```bash
docker compose -f docker-compose.external.yml ps
docker compose -f docker-compose.external.yml logs -f api
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health/ai-models
```

Browser: http://127.0.0.1:8000/docs

Test login:

```bash
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"student","password":"password"}'
```

---

## Step 7 — Run the frontend

Clone `ai-tutor-frontend` and set `.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
VITE_VOICE_URL=http://127.0.0.1:8000
```

If the browser is on another machine, use the server IP instead of `127.0.0.1`.

```bash
npm install
npm run dev
```

---

## Daily commands

| Action | Command |
|--------|---------|
| Start | `API_IMAGE=manithmettu/ai-tutor-backend:latest ENV_FILE=~/ai-tutor.env docker compose -f docker-compose.external.yml up -d` |
| Stop | `docker compose -f docker-compose.external.yml down` |
| Logs | `docker compose -f docker-compose.external.yml logs -f api` |
| Update image | `docker compose -f docker-compose.external.yml pull && docker compose -f docker-compose.external.yml up -d` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pull access denied` | Run `docker login` (private repo) or check image name |
| API keeps restarting | `docker compose -f docker-compose.external.yml logs api` — often DB connection failed |
| Cannot connect to Postgres | Cloud firewall must allow **your IP** on port **5434** |
| Out of memory | Give Docker **8 GB+ RAM** in Docker Desktop settings |
| Port 8000 in use | `API_PORT=8001` before `docker compose up` |

---

## What runs where

| Service | Where |
|---------|--------|
| API | Docker image from Hub |
| Redis | Docker container (automatic) |
| Postgres | **Cloud** (from your `DATABASE_URL`) |
| Uploads / models cache | Docker volumes on your machine |

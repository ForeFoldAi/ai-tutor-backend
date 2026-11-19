# Publish to Docker Hub (build locally → test → push)

## Prerequisites

- Docker Desktop or Docker Engine running
- [Docker Hub](https://hub.docker.com/) account
- `.env` in `ai-tutor-backend/` with `MISTRAL_API_KEY`, `JWT_SECRET_KEY`, `HF_TOKEN`

## 1. Build the image locally

```bash
cd ai-tutor-backend

export DOCKERHUB_USER=yourdockerhubuser

docker build -t ai-tutor-backend:latest .
docker tag ai-tutor-backend:latest ${DOCKERHUB_USER}/ai-tutor-backend:latest
```

## 2. Test locally (before pushing)

Use your existing `.env` and the pre-built image (no rebuild):

```bash
cd ai-tutor-backend

export DOCKERHUB_USER=yourdockerhubuser
export API_IMAGE=${DOCKERHUB_USER}/ai-tutor-backend:latest
export ENV_FILE="$(pwd)/.env"

docker compose -f docker-compose.image.yml up -d
```

Wait for startup (first run can take a few minutes), then check:

```bash
docker compose -f docker-compose.image.yml ps
docker compose -f docker-compose.image.yml logs -f api
curl http://127.0.0.1:8000/health/ai-models
```

Open http://127.0.0.1:8000/docs and try an endpoint if you want.

**Stop test stack when done:**

```bash
docker compose -f docker-compose.image.yml down
```

> `down` keeps database/upload volumes. Add `-v` only if you want a completely fresh test DB.

## 3. Push to Docker Hub

```bash
docker login

docker push ${DOCKERHUB_USER}/ai-tutor-backend:latest
```

Optional version tag (recommended for releases):

```bash
export VERSION=1.0.0
docker tag ai-tutor-backend:latest ${DOCKERHUB_USER}/ai-tutor-backend:${VERSION}
docker push ${DOCKERHUB_USER}/ai-tutor-backend:${VERSION}
docker push ${DOCKERHUB_USER}/ai-tutor-backend:latest
```

## 4. Tell your team

Image: `yourdockerhubuser/ai-tutor-backend:latest`  
Setup: see `deploy/TEAM_SETUP.md`

```bash
API_IMAGE=yourdockerhubuser/ai-tutor-backend:latest \
ENV_FILE=/path/to/their.env \
docker compose -f deploy/docker-compose.yml up -d
```

## Multi-arch (optional)

If teammates use both Apple Silicon and Linux x86:

```bash
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ${DOCKERHUB_USER}/ai-tutor-backend:latest \
  --push .
```

Test on your machine first with a single-arch local build; use buildx only when ready to publish for all platforms.

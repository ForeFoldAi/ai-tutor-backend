# Multi-replica API (Compose)

Use only after opting into **both** shared backends:

```bash
STORAGE_BACKEND=s3
VECTOR_BACKEND=qdrant
CELERY_AUTOSTART=false
```

## Why

Local `uploads_data` / `chroma_data` volumes are single-node. With MinIO (or AWS S3) and Qdrant, each API replica is interchangeable for REST + RAG.

## Pattern

```yaml
# Example — scale API after shared storage/vectors are on
services:
  api:
    deploy:
      replicas: 2
    environment:
      STORAGE_BACKEND: s3
      VECTOR_BACKEND: qdrant
      S3_ENDPOINT_URL: http://minio:9000
      QDRANT_URL: http://qdrant:6333
      CELERY_AUTOSTART: "false"
    # When STORAGE_BACKEND=s3 and VECTOR_BACKEND=qdrant, uploads/chroma
    # volume mounts are optional (cache only).
  lesson_worker:
    # Scale workers independently; same S3 + Qdrant env
    deploy:
      replicas: 2
```

Put a reverse proxy (Caddy/nginx) in front with long WebSocket timeouts for `/ws/voice` and `/ws/events`.

## Still not covered by this cutover

- In-process `/ws/events` hub (needs Redis pub/sub for true multi-instance fanout)
- Process-local voice speaker enrollments
- Lesson-planner export files (still local disk)

## Checklist

1. MinIO bucket exists (`minio-init`) or AWS credentials work
2. `migrate_chroma_to_qdrant.py` or `reindex_all_to_vector_backend.py` completed
3. `python tests/test_qdrant_roundtrip.py` passes
4. Smoke: upload on replica A, chat retrieve on replica B
5. `CELERY_AUTOSTART=false` on all API replicas

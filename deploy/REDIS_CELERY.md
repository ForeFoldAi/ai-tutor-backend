# Redis + Celery (Coolify / production)

Tutor answers and student chat history are **not** stored in Redis.
There is **no** student LLM token metering in this codebase today.

## Feature matrix

| Feature | Redis | Celery | Notes |
|---------|-------|--------|-------|
| Auth / JWT | No | No | Postgres sessions |
| Catalog upload / embed | No | No | API `BackgroundTasks` |
| Tutor chat / stream / voice answers | No | No | Always hits LLM |
| Student chat history | No | No | Postgres `student_tutor_chats` |
| LIA live (events, twin, guidance) | No | No | Sync Postgres; fail-open |
| Voice learner profile | Soft | No | Fail-open convenience cache |
| Lesson planner | **Yes** | **Yes** | Job state, Pub/Sub, rate limit |
| Mail reminders | **Yes** | **Yes** | Beat + `mail` queue |
| LIA weekly/monthly summaries | Yes | Yes | Optional; needs worker on `lia` |

**Minimal Coolify (students + tutor + live LIA):** API + Postgres (+ optional S3/Qdrant). No Redis/Celery required.

**Full product (teachers + mail):** add Redis + dedicated Celery workers.

## Coolify env

```bash
REDIS_URL=redis://:PASSWORD@your-redis-host:6379/0
CELERY_AUTOSTART=false
```

Do **not** set any tutor answer cache flag (removed).

## Worker commands

Lesson (required for lesson planner):

```bash
celery -A app.core.celery_app.celery_app worker \
  -Q lesson-generate,lesson-export,lesson-autosave,lesson-regenerate \
  --concurrency=2 --loglevel=info
```

Mail + Beat (required for daily reminders):

```bash
celery -A app.core.celery_app.celery_app worker -Q mail --concurrency=1 --loglevel=info
celery -A app.core.celery_app.celery_app beat --loglevel=info
```

LIA period summaries (optional):

```bash
celery -A app.core.celery_app.celery_app worker -Q lia --concurrency=1 --loglevel=info
```

Use the same `REDIS_URL` / `DATABASE_URL` on API and all workers. Never enable `CELERY_AUTOSTART` on multiple API replicas.

## Remaining Redis callers (audit)

| Area | Path |
|------|------|
| Celery broker/backend | `app/core/celery_app.py` |
| Lesson job state / cancel / PubSub | `app/services/lesson_planner/redis/job_state.py` |
| Lesson rate limit | `app/services/lesson_planner/rate_limit.py` |
| Lesson checkpoints | `app/services/lesson_planner/checkpoint/store.py` |
| Lesson WS progress | `app/modules/teacher/lesson_planner/ws.py` |
| Mail / LIA tasks | `app/services/mail/tasks.py`, `learning_intelligence/workers/tasks.py` |
| Voice learner profile (fail-open) | `app/services/learner_profile.py` via `app/core/cache.get_redis` |

## Capacity (5k–10k registered users)

Student chat does not load Redis. Size Redis for Celery + lesson-planner job keys (~512MB–1GB, `noeviction`). Scale lesson workers by teacher queue depth, not student count.

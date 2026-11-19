# Monitoring environment variables

Set these on the **API / worker Coolify apps** (never in frontend bundles).

## Feature flag
- `MONITORING_ENABLED=true` — enables request metrics middleware + workers should run Beat tasks
- `ENVIRONMENT=production`
- `RELEASE_SHA` or `GIT_SHA` — release correlation

## Sentry Cloud
- `SENTRY_DSN`
- `SENTRY_TRACES_SAMPLE_RATE=0.05`
- `SENTRY_WEBHOOK_SECRET` — for `/auth/admin/monitoring/webhooks/sentry`

## Loki
- `LOKI_URL=http://loki:3100` — Grafana Loki base URL (Coolify service)

## Coolify (server-side only)
- `COOLIFY_URL`
- `COOLIFY_TOKEN`
- `COOLIFY_APP_UUID` — API
- `COOLIFY_WORKER_UUID`
- `COOLIFY_VOICE_UUID` or `COOLIFY_APP_UUID_VOICE`
- `COOLIFY_FRONTEND_UUID`
- `COOLIFY_SERVER_UUID` — optional; first server used if empty

## Ingest / alerts
- `MONITORING_INGEST_TOKEN` (falls back to `VOICE_INTERNAL_TOKEN`)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `SLACK_WEBHOOK_URL`
- SMTP settings already used by the app (`SMTP_*`, `ADMIN_NOTIFY_EMAIL`)

## Voice / website probes
- `VOICE_URL` — NestJS public (or internal) base for `/rtc/health`  
  Example: `https://api.voice.tutor.forefoldai.com`  
  If unset in production, voice shows **unknown** (not down). Do **not** rely on `localhost:8080` inside the API container.
- `FRONTEND_URL` — website probe  
  Example: `https://learnmastro.com`

## Rollback
1. Set `MONITORING_ENABLED=false`
2. `alembic downgrade 20261107_0051` (drops `monitoring` schema only)
3. Redeploy previous API image

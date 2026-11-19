# Monitoring rollback

## Disable without schema drop
1. Set `MONITORING_ENABLED=false` on API + worker Coolify apps.
2. Redeploy. Middleware becomes a no-op; Beat tasks still run but settings.enabled can be toggled off in UI/DB.

## Database rollback (drops monitoring schema only)
```bash
alembic downgrade 20261107_0051
```
This executes `DROP SCHEMA IF EXISTS monitoring CASCADE` and does **not** touch student/teacher tables.

## Frontend rollback
1. Remove the Monitor nav item from `app-nav-items.ts` (or hide via feature flag).
2. Redeploy frontend. Existing Admin Panel is unchanged.

## Image rollback
Redeploy the previous Coolify image tags for API/worker/voice/frontend.

## Secrets
Rotate `COOLIFY_TOKEN`, `SENTRY_DSN`, `MONITORING_INGEST_TOKEN`, Slack/Telegram credentials if exposure is suspected.

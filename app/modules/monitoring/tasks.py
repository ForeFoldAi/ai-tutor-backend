"""Celery tasks: alert evaluation, retention, Coolify/Sentry sync."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.monitoring import QUEUE_MONITORING
from app.modules.monitoring.alerts.notify import dispatch_alert_channels
from app.modules.monitoring.detection.engine import (
    compare_threshold,
    consecutive_breaches,
    ewma_anomaly,
    mad_zscore,
    should_fire_with_cooldown,
)
from app.modules.monitoring.models import (
    MonAlert,
    MonAlertRule,
    MonHealthCheck,
    MonIssueEvent,
    MonLogIndex,
    MonMetricAggregate,
    MonSettings,
)
from app.modules.monitoring.service import get_or_create_settings

logger = logging.getLogger(__name__)


@celery_app.task(name="monitoring.evaluate_alert_rules", queue=QUEUE_MONITORING, bind=True, max_retries=3)
def evaluate_alert_rules(self) -> dict:
    db = SessionLocal()
    fired = 0
    try:
        settings = get_or_create_settings(db)
        if not settings.enabled:
            return {"ok": True, "skipped": True}
        rules = list(db.scalars(select(MonAlertRule).where(MonAlertRule.enabled.is_(True))).all())
        now = datetime.now(UTC)
        for rule in rules:
            since = now - timedelta(seconds=max(60, rule.window_seconds))
            rows = list(
                db.scalars(
                    select(MonMetricAggregate)
                    .where(
                        MonMetricAggregate.metric_key == rule.metric_key,
                        MonMetricAggregate.bucket_start >= since,
                    )
                    .order_by(MonMetricAggregate.bucket_start.asc())
                ).all()
            )
            values = []
            for r in rows:
                if r.count and r.sum_value is not None:
                    values.append(float(r.sum_value) / max(int(r.count), 1))
                elif r.p95 is not None:
                    values.append(float(r.p95))
            if not values:
                continue

            last = values[-1]
            breached = False
            if rule.detection_mode == "ewma":
                breached, _, _ = ewma_anomaly(values)
            elif rule.detection_mode == "mad":
                breached, _ = mad_zscore(values)
            else:
                flags = [compare_threshold(v, rule.comparator, float(rule.threshold)) for v in values]
                breached = consecutive_breaches(flags, max(1, rule.consecutive_failures))

            if not breached:
                # Recovery: resolve open alerts for this rule fingerprint
                open_alerts = list(
                    db.scalars(
                        select(MonAlert).where(
                            MonAlert.rule_id == rule.id,
                            MonAlert.status.in_(("firing", "acknowledged")),
                        )
                    ).all()
                )
                for alert in open_alerts:
                    alert.status = "resolved"
                    alert.resolved_at = now
                    dispatch_alert_channels(
                        db,
                        channels=list(rule.channels or ["in_app"]),
                        title=f"Recovered: {rule.name}",
                        message=f"{rule.metric_key} recovered (value={last})",
                        severity="info",
                        settings_row=settings,
                    )
                db.commit()
                continue

            fingerprint = f"rule:{rule.id}:{rule.metric_key}"
            existing = db.scalar(
                select(MonAlert)
                .where(MonAlert.fingerprint == fingerprint, MonAlert.status.in_(("firing", "acknowledged")))
                .order_by(MonAlert.fired_at.desc())
            )
            if existing and not should_fire_with_cooldown(
                last_notified_at=existing.last_notified_at,
                now=now,
                cooldown_seconds=int(rule.cooldown_seconds or 900),
            ):
                continue

            if existing is None:
                existing = MonAlert(
                    rule_id=rule.id,
                    fingerprint=fingerprint,
                    title=rule.name,
                    severity=rule.severity,
                    status="firing",
                    service_key=rule.service_key,
                    environment=rule.environment,
                    value=last,
                    threshold=float(rule.threshold),
                    message=f"{rule.metric_key}={last} ({rule.comparator} {rule.threshold})",
                    fired_at=now,
                )
                db.add(existing)
                db.flush()

            existing.last_notified_at = now
            existing.value = last
            dispatch_alert_channels(
                db,
                channels=list(rule.channels or ["in_app", "email"]),
                title=existing.title,
                message=existing.message,
                severity=existing.severity,
                settings_row=settings,
            )
            db.commit()
            fired += 1
        return {"ok": True, "fired": fired}
    except Exception as exc:
        logger.exception("evaluate_alert_rules failed")
        db.rollback()
        raise self.retry(exc=exc, countdown=30)
    finally:
        db.close()


@celery_app.task(name="monitoring.run_retention", queue=QUEUE_MONITORING)
def run_retention() -> dict:
    db = SessionLocal()
    try:
        settings = get_or_create_settings(db)
        now = datetime.now(UTC)
        logs_cut = now - timedelta(days=max(1, settings.retention_days_logs))
        metrics_cut = now - timedelta(days=max(1, settings.retention_days_metrics))
        events_cut = now - timedelta(days=max(1, settings.retention_days_events))
        deleted = {
            "log_index": db.execute(delete(MonLogIndex).where(MonLogIndex.ts < logs_cut)).rowcount,
            "metrics": db.execute(
                delete(MonMetricAggregate).where(MonMetricAggregate.bucket_start < metrics_cut)
            ).rowcount,
            "issue_events": db.execute(
                delete(MonIssueEvent).where(MonIssueEvent.created_at < events_cut)
            ).rowcount,
            "health_checks": db.execute(
                delete(MonHealthCheck).where(MonHealthCheck.checked_at < events_cut)
            ).rowcount,
        }
        db.commit()
        return {"ok": True, "deleted": deleted}
    except Exception:
        logger.exception("retention failed")
        db.rollback()
        return {"ok": False}
    finally:
        db.close()


@celery_app.task(name="monitoring.probe_services", queue=QUEUE_MONITORING)
def probe_services() -> dict:
    from app.modules.monitoring.health import probe_all
    from app.modules.monitoring.models import MonHealthCheck

    db = SessionLocal()
    try:
        probes = probe_all()
        for key, probe in probes.items():
            db.add(
                MonHealthCheck(
                    service_key=key,
                    status=str(probe.get("status") or "unknown"),
                    latency_ms=probe.get("latency_ms"),
                    detail={k: v for k, v in probe.items() if k not in ("status", "latency_ms")},
                )
            )
        db.commit()
        return {"ok": True, "probes": {k: v.get("status") for k, v in probes.items()}}
    except Exception:
        logger.exception("probe_services failed")
        db.rollback()
        return {"ok": False}
    finally:
        db.close()

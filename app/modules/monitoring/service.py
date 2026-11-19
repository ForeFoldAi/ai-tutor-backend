"""Business logic for monitoring APIs."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.modules.monitoring.coolify import get_coolify_client
from app.modules.monitoring.detection.engine import fingerprint_error
from app.modules.monitoring.health import probe_all
from app.modules.monitoring.loki import get_loki_client
from app.modules.monitoring.models import (
    MonAlert,
    MonAlertAck,
    MonAlertRule,
    MonAuditEvent,
    MonDeployment,
    MonHealthCheck,
    MonIncident,
    MonIncidentComment,
    MonIncidentEvent,
    MonIssue,
    MonIssueComment,
    MonIssueEvent,
    MonLogIndex,
    MonMetricAggregate,
    MonService,
    MonSettings,
)
from app.modules.monitoring.redaction import preview_message, redact_obj
from app.modules.monitoring.schemas import (
    AlertRuleIn,
    IncidentCreate,
    IncidentPatch,
    IngestEventIn,
    IssuePatch,
    SettingsPatch,
    VoiceIngestIn,
)

logger = logging.getLogger(__name__)


def _env() -> str:
    return os.environ.get("ENVIRONMENT", "production")


def _ai_tutor_status(probes: dict[str, Any]) -> str:
    """Overview 'AI Tutor' card — API text tutor + voice, not voice-only."""
    api = str((probes.get("api") or {}).get("status") or "unknown")
    voice = str((probes.get("voice") or {}).get("status") or "unknown")
    if api == "down":
        return "down"
    if voice == "down":
        return "degraded"
    if api == "healthy" and voice in ("healthy", "unknown"):
        return "healthy" if voice == "healthy" else "healthy"
    if api == "healthy":
        return "healthy"
    return api or "unknown"


def get_or_create_settings(db: Session) -> MonSettings:
    row = db.get(MonSettings, 1)
    if row is None:
        row = MonSettings(id=1)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def audit(db: Session, *, actor_user_id: int | None, action: str, entity_type: str, entity_id: str | None, detail: dict | None = None) -> None:
    try:
        db.add(
            MonAuditEvent(
                actor_user_id=actor_user_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                detail=redact_obj(detail) if detail else None,
            )
        )
        db.commit()
    except Exception:
        db.rollback()


def _percentile_from_aggs(rows: list[MonMetricAggregate], attr: str) -> float | None:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_overview(
    db: Session,
    *,
    environment: str | None = None,
    service: str | None = None,
    hours: int = 24,
) -> dict[str, Any]:
    environment = environment or _env()
    since = datetime.now(UTC) - timedelta(hours=max(1, hours))
    unavailable: list[str] = []

    probes = probe_all()
    for key, probe in probes.items():
        try:
            db.add(
                MonHealthCheck(
                    service_key=key,
                    status=str(probe.get("status") or "unknown"),
                    latency_ms=probe.get("latency_ms"),
                    detail={k: v for k, v in probe.items() if k not in ("status", "latency_ms")},
                )
            )
        except Exception:
            pass
    try:
        db.commit()
    except Exception:
        db.rollback()

    open_issues = db.scalar(
        select(func.count()).select_from(MonIssue).where(MonIssue.status.in_(("open", "reopened")))
    ) or 0
    active_incidents = db.scalar(
        select(func.count()).select_from(MonIncident).where(MonIncident.status.in_(("open", "acknowledged", "investigating")))
    ) or 0
    affected_users = db.scalar(
        select(func.coalesce(func.sum(MonIssue.affected_users), 0)).where(MonIssue.status.in_(("open", "reopened")))
    ) or 0

    dur_q = select(MonMetricAggregate).where(
        MonMetricAggregate.metric_key == "http.request.duration_ms",
        MonMetricAggregate.bucket_start >= since,
        MonMetricAggregate.environment == environment,
    )
    if service:
        dur_q = dur_q.where(MonMetricAggregate.service_key == service)
    dur_rows = list(db.scalars(dur_q).all())

    count_q = select(func.coalesce(func.sum(MonMetricAggregate.sum_value), 0)).where(
        MonMetricAggregate.metric_key == "http.request.count",
        MonMetricAggregate.bucket_start >= since,
        MonMetricAggregate.environment == environment,
    )
    err_q = select(func.coalesce(func.sum(MonMetricAggregate.sum_value), 0)).where(
        MonMetricAggregate.metric_key == "http.request.errors",
        MonMetricAggregate.bucket_start >= since,
        MonMetricAggregate.environment == environment,
    )
    request_volume = int(db.scalar(count_q) or 0)
    error_count = int(db.scalar(err_q) or 0)
    error_rate = (error_count / request_volume) if request_volume else None
    success_rate = (1.0 - error_rate) if error_rate is not None else None

    coolify = get_coolify_client().infrastructure_snapshot()
    if coolify.get("unavailable"):
        unavailable.extend(list(coolify["unavailable"]))

    recent_deployments = [
        {
            "id": d.id,
            "service_key": d.service_key,
            "status": d.status,
            "version": d.version,
            "started_at": d.started_at.isoformat(),
        }
        for d in db.scalars(select(MonDeployment).order_by(desc(MonDeployment.started_at)).limit(5)).all()
    ]
    if not recent_deployments and coolify.get("deployments"):
        recent_deployments = coolify["deployments"][:5]

    recent_alerts = [
        {
            "id": a.id,
            "title": a.title,
            "severity": a.severity,
            "status": a.status,
            "fired_at": a.fired_at.isoformat(),
        }
        for a in db.scalars(select(MonAlert).order_by(desc(MonAlert.fired_at)).limit(8)).all()
    ]

    services = [
        {"key": s.key, "name": s.name, "kind": s.kind, "status": (probes.get(s.key) or {}).get("status", "unknown")}
        for s in db.scalars(select(MonService).where(MonService.is_active.is_(True))).all()
    ]

    statuses = [p.get("status") for p in probes.values()]
    if any(s == "down" for s in statuses) or active_incidents:
        overall = "critical"
    elif any(s == "degraded" for s in statuses) or open_issues:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "overall_status": overall,
        "api_success_rate": success_rate,
        "error_rate": error_rate,
        "request_volume": request_volume,
        "latency_p50_ms": _percentile_from_aggs(dur_rows, "p50"),
        "latency_p95_ms": _percentile_from_aggs(dur_rows, "p95"),
        "latency_p99_ms": _percentile_from_aggs(dur_rows, "p99"),
        "active_incidents": int(active_incidents),
        "open_issues": int(open_issues),
        "affected_users": int(affected_users),
        "cpu_percent": coolify.get("cpu_percent"),
        "ram_percent": coolify.get("ram_percent"),
        "disk_percent": coolify.get("disk_percent"),
        "database_status": (probes.get("postgres") or {}).get("status", "unknown"),
        "redis_status": (probes.get("redis") or {}).get("status", "unknown"),
        "containers": coolify.get("containers") or [],
        # Text tutor rides the API; voice is separate. Overall AI stack:
        # down only if API is down; degraded if voice is down; else healthy/unknown.
        "ai_tutor_status": _ai_tutor_status(probes),
        "recent_deployments": recent_deployments,
        "recent_alerts": recent_alerts,
        "services": services,
        "unavailable": unavailable,
    }


def list_issues(
    db: Session,
    *,
    status: str | None = None,
    severity: str | None = None,
    service: str | None = None,
    q: str | None = None,
    page: int = 1,
    page_size: int = 25,
) -> tuple[list[MonIssue], int]:
    stmt = select(MonIssue)
    count_stmt = select(func.count()).select_from(MonIssue)
    if status:
        stmt = stmt.where(MonIssue.status == status)
        count_stmt = count_stmt.where(MonIssue.status == status)
    if severity:
        stmt = stmt.where(MonIssue.severity == severity)
        count_stmt = count_stmt.where(MonIssue.severity == severity)
    if service:
        stmt = stmt.where(MonIssue.service_key == service)
        count_stmt = count_stmt.where(MonIssue.service_key == service)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(MonIssue.title.ilike(like))
        count_stmt = count_stmt.where(MonIssue.title.ilike(like))
    total = int(db.scalar(count_stmt) or 0)
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items = list(
        db.scalars(
            stmt.order_by(desc(MonIssue.last_seen_at)).offset((page - 1) * page_size).limit(page_size)
        ).all()
    )
    return items, total


def get_issue_detail(db: Session, issue_id: int) -> dict[str, Any] | None:
    issue = db.get(MonIssue, issue_id)
    if not issue:
        return None
    events = list(
        db.scalars(
            select(MonIssueEvent).where(MonIssueEvent.issue_id == issue_id).order_by(desc(MonIssueEvent.created_at)).limit(50)
        ).all()
    )
    comments = list(
        db.scalars(
            select(MonIssueComment)
            .where(MonIssueComment.issue_id == issue_id)
            .order_by(MonIssueComment.created_at.asc())
        ).all()
    )
    return {
        "issue": issue,
        "events": [
            {
                "id": e.id,
                "event_id": e.event_id,
                "message": e.message,
                "stacktrace": e.stacktrace,
                "breadcrumbs": e.breadcrumbs,
                "request": e.request,
                "user_ref": e.user_ref,
                "request_id": e.request_id,
                "trace_id": e.trace_id,
                "release": e.release,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
        "comments": comments,
    }


def patch_issue(db: Session, issue_id: int, patch: IssuePatch, actor_user_id: int) -> MonIssue | None:
    issue = db.get(MonIssue, issue_id)
    if not issue:
        return None
    if patch.status is not None:
        issue.status = patch.status
        if patch.status == "resolved":
            issue.resolved_at = datetime.now(UTC)
        elif patch.status in ("open", "reopened"):
            issue.resolved_at = None
    if patch.severity is not None:
        issue.severity = patch.severity
    if patch.assignee_user_id is not None:
        issue.assignee_user_id = patch.assignee_user_id
    issue.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(issue)
    audit(db, actor_user_id=actor_user_id, action="issue.patch", entity_type="issue", entity_id=str(issue_id), detail=patch.model_dump())
    return issue


def add_issue_comment(db: Session, issue_id: int, author_user_id: int, body: str) -> MonIssueComment | None:
    if not db.get(MonIssue, issue_id):
        return None
    row = MonIssueComment(issue_id=issue_id, author_user_id=author_user_id, body=body.strip())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def ingest_event(db: Session, payload: IngestEventIn) -> MonIssue:
    fp = payload.fingerprint or fingerprint_error(
        title=payload.title,
        culprit=(payload.meta or {}).get("culprit") if payload.meta else None,
        service=payload.service,
        environment=payload.environment,
    )
    now = datetime.now(UTC)
    issue = db.scalar(select(MonIssue).where(MonIssue.fingerprint == fp))
    if issue is None:
        issue = MonIssue(
            fingerprint=fp,
            title=payload.title[:500],
            culprit=((payload.meta or {}).get("culprit") if payload.meta else None),
            severity=payload.severity,
            status="open",
            environment=payload.environment,
            service_key=payload.service,
            release=payload.release,
            occurrence_count=1,
            affected_users=1 if payload.user_ref else 0,
            first_seen_at=now,
            last_seen_at=now,
            meta=redact_obj(payload.meta) if payload.meta else None,
        )
        db.add(issue)
        db.flush()
    else:
        issue.occurrence_count = int(issue.occurrence_count or 0) + 1
        issue.last_seen_at = now
        issue.updated_at = now
        if issue.status == "resolved":
            issue.status = "reopened"
            issue.resolved_at = None
        if payload.user_ref:
            issue.affected_users = int(issue.affected_users or 0) + 1
        if payload.release:
            issue.release = payload.release

    db.add(
        MonIssueEvent(
            issue_id=issue.id,
            message=preview_message(payload.message or payload.title),
            stacktrace=redact_obj(payload.stacktrace) if payload.stacktrace else None,
            breadcrumbs=redact_obj(payload.breadcrumbs) if payload.breadcrumbs else None,
            request=redact_obj(payload.request) if payload.request else None,
            user_ref=payload.user_ref,
            request_id=payload.request_id,
            trace_id=payload.trace_id,
            release=payload.release,
            environment=payload.environment,
        )
    )
    db.add(
        MonLogIndex(
            ts=now,
            environment=payload.environment,
            service=payload.service,
            module="ingest",
            severity=payload.severity,
            message_preview=preview_message(payload.message or payload.title),
            request_id=payload.request_id,
            trace_id=payload.trace_id,
            release=payload.release,
            duration_ms=payload.duration_ms,
        )
    )
    db.commit()
    db.refresh(issue)
    return issue


def query_logs(
    db: Session,
    *,
    q: str | None = None,
    service: str | None = None,
    severity: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    hours: int = 6,
    limit: int = 100,
) -> dict[str, Any]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=max(1, hours))
    loki = get_loki_client()
    label_parts = ['{job=~".+"}']
    if service:
        label_parts = [f'{{service="{service}"}}']
    query = label_parts[0]
    if q:
        query = f'{query} |= `{q}`'
    if severity:
        query = f'{query} |= `"severity":"{severity}"`'
    if request_id:
        query = f'{query} |= `{request_id}`'
    if trace_id:
        query = f'{query} |= `{trace_id}`'

    loki_result = loki.query_range(query=query, start=start, end=end, limit=limit)
    items: list[dict[str, Any]] = []
    if not loki_result.get("unavailable"):
        for entry in loki_result.get("result") or []:
            ts = None
            try:
                ts = datetime.fromtimestamp(int(entry["ts_ns"]) / 1e9, tz=UTC)
            except Exception:
                ts = None
            items.append(
                {
                    "ts": ts,
                    "environment": entry.get("environment"),
                    "service": entry.get("service"),
                    "module": entry.get("module"),
                    "severity": entry.get("severity"),
                    "message": entry.get("message") or "",
                    "request_id": entry.get("request_id"),
                    "trace_id": entry.get("trace_id"),
                    "release": entry.get("release"),
                    "duration_ms": entry.get("duration_ms"),
                    "raw": entry.get("raw"),
                }
            )
        return {"items": items, "source": "loki", "unavailable": False, "message": None, "next_cursor": None}

    # Fallback: metadata index in Postgres (previews only)
    stmt = select(MonLogIndex).where(MonLogIndex.ts >= start)
    if service:
        stmt = stmt.where(MonLogIndex.service == service)
    if severity:
        stmt = stmt.where(MonLogIndex.severity == severity)
    if request_id:
        stmt = stmt.where(MonLogIndex.request_id == request_id)
    if trace_id:
        stmt = stmt.where(MonLogIndex.trace_id == trace_id)
    if q:
        stmt = stmt.where(MonLogIndex.message_preview.ilike(f"%{q}%"))
    rows = list(db.scalars(stmt.order_by(desc(MonLogIndex.ts)).limit(limit)).all())
    items = [
        {
            "ts": r.ts,
            "environment": r.environment,
            "service": r.service,
            "module": r.module,
            "severity": r.severity,
            "message": r.message_preview,
            "request_id": r.request_id,
            "trace_id": r.trace_id,
            "release": r.release,
            "duration_ms": r.duration_ms,
            "raw": None,
        }
        for r in rows
    ]
    return {
        "items": items,
        "source": "postgres_index",
        "unavailable": True,
        "message": loki_result.get("message") or "Loki unavailable; showing metadata index only",
        "next_cursor": None,
    }


def list_incidents(db: Session, *, status: str | None = None) -> tuple[list[MonIncident], int]:
    stmt = select(MonIncident)
    count_stmt = select(func.count()).select_from(MonIncident)
    if status:
        stmt = stmt.where(MonIncident.status == status)
        count_stmt = count_stmt.where(MonIncident.status == status)
    total = int(db.scalar(count_stmt) or 0)
    items = list(db.scalars(stmt.order_by(desc(MonIncident.opened_at)).limit(100)).all())
    return items, total


def create_incident(db: Session, payload: IncidentCreate, actor_user_id: int) -> MonIncident:
    now = datetime.now(UTC)
    inc = MonIncident(
        title=payload.title,
        severity=payload.severity,
        status="open",
        owner_user_id=actor_user_id,
        environment=payload.environment,
        service_key=payload.service_key,
        related_issue_ids=payload.related_issue_ids,
        opened_at=now,
    )
    db.add(inc)
    db.flush()
    db.add(
        MonIncidentEvent(
            incident_id=inc.id,
            kind="created",
            message="Incident opened",
            actor_user_id=actor_user_id,
        )
    )
    db.commit()
    db.refresh(inc)
    audit(db, actor_user_id=actor_user_id, action="incident.create", entity_type="incident", entity_id=str(inc.id))
    return inc


def get_incident_detail(db: Session, incident_id: int) -> dict[str, Any] | None:
    inc = db.get(MonIncident, incident_id)
    if not inc:
        return None
    events = list(
        db.scalars(
            select(MonIncidentEvent)
            .where(MonIncidentEvent.incident_id == incident_id)
            .order_by(MonIncidentEvent.created_at.asc())
        ).all()
    )
    comments = list(
        db.scalars(
            select(MonIncidentComment)
            .where(MonIncidentComment.incident_id == incident_id)
            .order_by(MonIncidentComment.created_at.asc())
        ).all()
    )
    return {
        "incident": inc,
        "events": [
            {
                "id": e.id,
                "kind": e.kind,
                "message": e.message,
                "actor_user_id": e.actor_user_id,
                "meta": e.meta,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
        "comments": [
            {
                "id": c.id,
                "author_user_id": c.author_user_id,
                "body": c.body,
                "internal": c.internal,
                "created_at": c.created_at.isoformat(),
            }
            for c in comments
        ],
    }


def patch_incident(db: Session, incident_id: int, patch: IncidentPatch, actor_user_id: int) -> MonIncident | None:
    inc = db.get(MonIncident, incident_id)
    if not inc:
        return None
    now = datetime.now(UTC)
    if patch.status is not None:
        prev = inc.status
        inc.status = patch.status
        if patch.status == "acknowledged" and not inc.acknowledged_at:
            inc.acknowledged_at = now
        if patch.status == "resolved":
            inc.resolved_at = now
        db.add(
            MonIncidentEvent(
                incident_id=inc.id,
                kind="status_change",
                message=f"{prev} -> {patch.status}",
                actor_user_id=actor_user_id,
            )
        )
    if patch.severity is not None:
        inc.severity = patch.severity
    if patch.owner_user_id is not None:
        inc.owner_user_id = patch.owner_user_id
    if patch.root_cause is not None:
        inc.root_cause = patch.root_cause
    if patch.postmortem is not None:
        inc.postmortem = patch.postmortem
    inc.updated_at = now
    db.commit()
    db.refresh(inc)
    return inc


def add_incident_comment(
    db: Session, incident_id: int, author_user_id: int, body: str, *, internal: bool = True
) -> MonIncidentComment | None:
    if not db.get(MonIncident, incident_id):
        return None
    row = MonIncidentComment(
        incident_id=incident_id, author_user_id=author_user_id, body=body.strip(), internal=internal
    )
    db.add(row)
    db.add(
        MonIncidentEvent(
            incident_id=incident_id,
            kind="comment",
            message="Comment added",
            actor_user_id=author_user_id,
        )
    )
    db.commit()
    db.refresh(row)
    return row


def list_alert_rules(db: Session) -> list[MonAlertRule]:
    return list(db.scalars(select(MonAlertRule).order_by(MonAlertRule.id.asc())).all())


def create_alert_rule(db: Session, payload: AlertRuleIn, actor_user_id: int) -> MonAlertRule:
    row = MonAlertRule(**payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    audit(db, actor_user_id=actor_user_id, action="alert_rule.create", entity_type="alert_rule", entity_id=str(row.id))
    return row


def list_alerts(db: Session, *, status: str | None = None) -> tuple[list[MonAlert], int]:
    stmt = select(MonAlert)
    count_stmt = select(func.count()).select_from(MonAlert)
    if status:
        stmt = stmt.where(MonAlert.status == status)
        count_stmt = count_stmt.where(MonAlert.status == status)
    total = int(db.scalar(count_stmt) or 0)
    items = list(db.scalars(stmt.order_by(desc(MonAlert.fired_at)).limit(100)).all())
    return items, total


def ack_alert(db: Session, alert_id: int, user_id: int, note: str | None = None) -> MonAlert | None:
    alert = db.get(MonAlert, alert_id)
    if not alert:
        return None
    alert.status = "acknowledged"
    db.add(MonAlertAck(alert_id=alert_id, user_id=user_id, note=note))
    db.commit()
    db.refresh(alert)
    return alert


def list_deployments(db: Session) -> list[MonDeployment]:
    rows = list(db.scalars(select(MonDeployment).order_by(desc(MonDeployment.started_at)).limit(50)).all())
    if rows:
        return rows
    # Sync lightweight view from Coolify without fabricating metrics
    snap = get_coolify_client().infrastructure_snapshot()
    out: list[MonDeployment] = []
    for d in snap.get("deployments") or []:
        out.append(
            MonDeployment(
                id=0,
                service_key=str(d.get("service_key") or "api"),
                environment=_env(),
                version=None,
                commit_sha=str(d.get("commit") or "") or None,
                coolify_deployment_id=str(d.get("id") or "") or None,
                status=str(d.get("status") or "unknown"),
                started_at=datetime.now(UTC),
            )
        )
    return out


def infrastructure(db: Session) -> dict[str, Any]:
    snap = get_coolify_client().infrastructure_snapshot()
    probes = probe_all()
    snap["probes"] = probes
    return snap


def performance(db: Session, *, hours: int = 24, service: str | None = None) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=max(1, hours))
    stmt = select(MonMetricAggregate).where(
        MonMetricAggregate.metric_key == "http.request.duration_ms",
        MonMetricAggregate.bucket_start >= since,
    )
    if service:
        stmt = stmt.where(MonMetricAggregate.service_key == service)
    rows = list(db.scalars(stmt.order_by(MonMetricAggregate.bucket_start.asc())).all())
    series = [
        {
            "ts": r.bucket_start.isoformat(),
            "p50": r.p50,
            "p95": r.p95,
            "p99": r.p99,
            "count": r.count,
        }
        for r in rows
    ]
    slow = list(
        db.scalars(
            select(MonLogIndex)
            .where(MonLogIndex.duration_ms.is_not(None), MonLogIndex.duration_ms >= 2000, MonLogIndex.ts >= since)
            .order_by(desc(MonLogIndex.duration_ms))
            .limit(20)
        ).all()
    )
    return {
        "series": series,
        "p50_ms": _percentile_from_aggs(rows, "p50"),
        "p95_ms": _percentile_from_aggs(rows, "p95"),
        "p99_ms": _percentile_from_aggs(rows, "p99"),
        "slow_requests": [
            {
                "ts": s.ts.isoformat(),
                "message": s.message_preview,
                "duration_ms": s.duration_ms,
                "request_id": s.request_id,
                "service": s.service,
            }
            for s in slow
        ],
        "unavailable": [] if rows else ["metric_aggregates"],
    }


def ai_tutor_metrics(db: Session, *, hours: int = 24) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=max(1, hours))
    unavailable: list[str] = []
    token_prompt = 0
    token_completion = 0
    try:
        from app.modules.users.models import LlmUsage

        rows = db.execute(
            select(
                func.coalesce(func.sum(LlmUsage.prompt_tokens), 0),
                func.coalesce(func.sum(LlmUsage.completion_tokens), 0),
                func.count(),
            ).where(LlmUsage.created_at >= since)
        ).one()
        token_prompt = int(rows[0] or 0)
        token_completion = int(rows[1] or 0)
        volume = int(rows[2] or 0)
    except Exception:
        volume = 0
        unavailable.append("llm_usage")

    voice_probe = probe_all().get("voice") or {}
    qdrant = probe_all().get("qdrant") or {}

    err_count = int(
        db.scalar(
            select(func.count())
            .select_from(MonLogIndex)
            .where(
                MonLogIndex.ts >= since,
                MonLogIndex.severity.in_(("error", "critical")),
                MonLogIndex.service.in_(("api", "voice")),
                MonLogIndex.module.in_(("rag", "voice", "chat", "ingest")),
            )
        )
        or 0
    )

    routing: dict[str, Any] = {}
    try:
        from app.services.llm_client import routing_snapshot

        routing = routing_snapshot()
    except Exception as exc:
        unavailable.append("llm_routing")
        routing = {"error": str(exc)}

    active: dict[str, Any] = {}
    try:
        from app.modules.monitoring.activity import activity_snapshot

        active = activity_snapshot()
    except Exception as exc:
        unavailable.append("active_users")
        active = {"error": str(exc)}

    return {
        "request_volume": volume,
        "success_rate": None if volume == 0 else max(0.0, 1.0 - (err_count / max(volume, 1))),
        "latency_p95_ms": None,
        "timeouts": 0,
        "token_prompt": token_prompt,
        "token_completion": token_completion,
        "model_errors": err_count,
        "rate_limits": 0,
        "rag_failures": 0,
        "voice": voice_probe,
        "vector_status": qdrant.get("status", "unknown"),
        "routing": routing,
        "active_users": active,
        "series": [],
        "unavailable": unavailable,
    }


def llm_routing(db: Session | None = None) -> dict[str, Any]:
    """Dedicated endpoint payload for primary/secondary + API key status."""
    del db  # reserved for future persistence
    try:
        from app.services.llm_client import routing_snapshot

        return routing_snapshot()
    except Exception as exc:
        return {
            "primary_provider": None,
            "secondary_provider": None,
            "features": [],
            "last_used": {},
            "unavailable": ["llm_routing"],
            "error": str(exc),
        }


def active_users(*, minutes: int = 10) -> dict[str, Any]:
    from app.modules.monitoring.activity import activity_snapshot

    return activity_snapshot(active_seconds=max(1, minutes) * 60)


def ingest_voice(db: Session, payload: VoiceIngestIn) -> dict[str, Any]:
    """Store voice metrics aggregates only — never transcripts/audio."""
    now = datetime.now(UTC)
    bucket = now.replace(second=0, microsecond=0)
    env = _env()
    mapping = {
        "voice.stt_ms": payload.stt_ms,
        "voice.tts_ms": payload.tts_ms,
        "voice.rag_ms": payload.rag_ms,
        "voice.ttfa_ms": payload.ttfa_ms,
        "voice.total_ms": payload.total_ms,
    }
    for key, value in mapping.items():
        if value is None:
            continue
        _bump_metric(db, bucket, key, float(value), service_key="voice", environment=env)
    if payload.error:
        ingest_event(
            db,
            IngestEventIn(
                source="voice",
                service="voice",
                environment=env,
                severity="error",
                title=f"voice:{payload.event}",
                message=preview_message(payload.error),
                meta={"event": payload.event, "session_id": payload.session_id},
            ),
        )
    else:
        db.commit()
    return {"ok": True}


def _bump_metric(
    db: Session,
    bucket: datetime,
    metric_key: str,
    value: float,
    *,
    service_key: str,
    environment: str,
) -> None:
    row = db.execute(
        select(MonMetricAggregate).where(
            MonMetricAggregate.bucket_start == bucket,
            MonMetricAggregate.metric_key == metric_key,
            MonMetricAggregate.service_key == service_key,
            MonMetricAggregate.environment == environment,
        )
    ).scalar_one_or_none()
    if row is None:
        db.add(
            MonMetricAggregate(
                bucket_start=bucket,
                metric_key=metric_key,
                service_key=service_key,
                environment=environment,
                count=1,
                sum_value=value,
                min_value=value,
                max_value=value,
                p50=value,
                p95=value,
                p99=value,
            )
        )
    else:
        row.count += 1
        row.sum_value += value
        row.min_value = value if row.min_value is None else min(row.min_value, value)
        row.max_value = value if row.max_value is None else max(row.max_value, value)
        avg = row.sum_value / max(row.count, 1)
        row.p50 = avg
        row.p95 = row.max_value
        row.p99 = row.max_value


def ux_metrics(db: Session, *, hours: int = 24) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=max(1, hours))
    rows = list(
        db.scalars(
            select(MonMetricAggregate)
            .where(
                MonMetricAggregate.metric_key.like("web_vitals.%"),
                MonMetricAggregate.bucket_start >= since,
            )
            .order_by(MonMetricAggregate.bucket_start.asc())
        ).all()
    )
    web_vitals: dict[str, Any] = {}
    for r in rows:
        web_vitals[r.metric_key] = {"p50": r.p50, "p95": r.p95, "count": r.count}
    return {
        "web_vitals": web_vitals,
        "series": [
            {"ts": r.bucket_start.isoformat(), "metric": r.metric_key, "p75": r.p95, "count": r.count}
            for r in rows
        ],
        "unavailable": [] if rows else ["web_vitals"],
    }


def reports(db: Session, *, hours: int = 168) -> dict[str, Any]:
    overview = build_overview(db, hours=hours)
    return {
        "summary": {
            "overall_status": overview["overall_status"],
            "request_volume": overview["request_volume"],
            "error_rate": overview["error_rate"],
            "open_issues": overview["open_issues"],
            "active_incidents": overview["active_incidents"],
        },
        "series": [],
    }


def get_settings_out(db: Session) -> dict[str, Any]:
    row = get_or_create_settings(db)
    return {
        "retention_days_logs": row.retention_days_logs,
        "retention_days_metrics": row.retention_days_metrics,
        "retention_days_events": row.retention_days_events,
        "auto_refresh_seconds": row.auto_refresh_seconds,
        "telegram_chat_id": row.telegram_chat_id,
        "slack_webhook_configured": bool(row.slack_webhook_url),
        "alert_email_to": row.alert_email_to,
        "enabled": row.enabled,
        "meta": row.meta,
    }


def patch_settings(db: Session, patch: SettingsPatch, actor_user_id: int) -> dict[str, Any]:
    row = get_or_create_settings(db)
    data = patch.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(row, k, v)
    row.updated_at = datetime.now(UTC)
    db.commit()
    audit(db, actor_user_id=actor_user_id, action="settings.patch", entity_type="settings", entity_id="1")
    return get_settings_out(db)


def list_services(db: Session) -> list[dict[str, Any]]:
    probes = probe_all()
    return [
        {
            "id": s.id,
            "key": s.key,
            "name": s.name,
            "kind": s.kind,
            "environment": s.environment,
            "status": (probes.get(s.key) or {}).get("status", "unknown"),
            "latency_ms": (probes.get(s.key) or {}).get("latency_ms"),
        }
        for s in db.scalars(select(MonService).where(MonService.is_active.is_(True))).all()
    ]

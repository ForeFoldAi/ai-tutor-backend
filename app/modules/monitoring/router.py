"""Master Admin monitoring API — /auth/admin/monitoring/*."""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.monitoring import service as svc
from app.modules.monitoring.schemas import (
    AlertListResponse,
    AlertOut,
    AlertRuleIn,
    AlertRuleOut,
    AiTutorMetricsResponse,
    DeploymentOut,
    IncidentCommentIn,
    IncidentCreate,
    IncidentDetailResponse,
    IncidentListResponse,
    IncidentOut,
    IncidentPatch,
    InfrastructureResponse,
    IngestEventIn,
    IssueCommentIn,
    IssueCommentOut,
    IssueDetailResponse,
    IssueListResponse,
    IssueOut,
    IssuePatch,
    LogListResponse,
    OverviewResponse,
    PerformanceResponse,
    ReportsResponse,
    SettingsOut,
    SettingsPatch,
    UxMetricsResponse,
    VoiceIngestIn,
)
from app.modules.users.models import User

router = APIRouter(
    prefix="/auth/admin/monitoring",
    tags=["master-admin-monitoring"],
)


def _internal_token_ok(header: str | None) -> bool:
    expected = os.environ.get("MONITORING_INGEST_TOKEN") or os.environ.get("VOICE_INTERNAL_TOKEN") or ""
    return bool(expected) and header == expected


@router.get("/overview", response_model=OverviewResponse)
def overview(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    environment: str | None = Query(default="production"),
    service: str | None = None,
    hours: int = Query(default=24, ge=1, le=720),
) -> Any:
    return svc.build_overview(db, environment=environment, service=service, hours=hours)


@router.get("/issues", response_model=IssueListResponse)
def issues(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    status_filter: str | None = Query(default=None, alias="status"),
    severity: str | None = None,
    service: str | None = None,
    q: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> Any:
    items, total = svc.list_issues(
        db, status=status_filter, severity=severity, service=service, q=q, page=page, page_size=page_size
    )
    return {
        "items": [IssueOut.model_validate(i) for i in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/issues/{issue_id}", response_model=IssueDetailResponse)
def issue_detail(
    issue_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    detail = svc.get_issue_detail(db, issue_id)
    if not detail:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Issue not found")
    return {
        "issue": IssueOut.model_validate(detail["issue"]),
        "events": detail["events"],
        "comments": [IssueCommentOut.model_validate(c) for c in detail["comments"]],
    }


@router.patch("/issues/{issue_id}", response_model=IssueOut)
def issue_patch(
    issue_id: int,
    body: IssuePatch,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    row = svc.patch_issue(db, issue_id, body, int(user.id))
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Issue not found")
    return IssueOut.model_validate(row)


@router.post("/issues/{issue_id}/comments", response_model=IssueCommentOut)
def issue_comment(
    issue_id: int,
    body: IssueCommentIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    row = svc.add_issue_comment(db, issue_id, int(user.id), body.body)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Issue not found")
    return IssueCommentOut.model_validate(row)


@router.get("/logs", response_model=LogListResponse)
def logs(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    q: str | None = None,
    service: str | None = None,
    severity: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    hours: int = Query(default=6, ge=1, le=168),
    limit: int = Query(default=100, ge=1, le=500),
) -> Any:
    return svc.query_logs(
        db,
        q=q,
        service=service,
        severity=severity,
        request_id=request_id,
        trace_id=trace_id,
        hours=hours,
        limit=limit,
    )


@router.get("/logs/export")
def logs_export(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    q: str | None = None,
    service: str | None = None,
    hours: int = Query(default=6, ge=1, le=168),
) -> PlainTextResponse:
    result = svc.query_logs(db, q=q, service=service, hours=hours, limit=500)
    lines = []
    for item in result.get("items") or []:
        lines.append(
            f"{item.get('ts')}\t{item.get('severity')}\t{item.get('service')}\t{item.get('message')}"
        )
    return PlainTextResponse("\n".join(lines), media_type="text/plain")


@router.get("/incidents", response_model=IncidentListResponse)
def incidents(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    status_filter: str | None = Query(default=None, alias="status"),
) -> Any:
    items, total = svc.list_incidents(db, status=status_filter)
    return {"items": [IncidentOut.model_validate(i) for i in items], "total": total}


@router.post("/incidents", response_model=IncidentOut)
def incident_create(
    body: IncidentCreate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return IncidentOut.model_validate(svc.create_incident(db, body, int(user.id)))


@router.get("/incidents/{incident_id}", response_model=IncidentDetailResponse)
def incident_detail(
    incident_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    detail = svc.get_incident_detail(db, incident_id)
    if not detail:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return {
        "incident": IncidentOut.model_validate(detail["incident"]),
        "events": detail["events"],
        "comments": detail["comments"],
    }


@router.patch("/incidents/{incident_id}", response_model=IncidentOut)
def incident_patch(
    incident_id: int,
    body: IncidentPatch,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    row = svc.patch_incident(db, incident_id, body, int(user.id))
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return IncidentOut.model_validate(row)


@router.post("/incidents/{incident_id}/comments")
def incident_comment(
    incident_id: int,
    body: IncidentCommentIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    row = svc.add_incident_comment(db, incident_id, int(user.id), body.body, internal=body.internal)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return {
        "id": row.id,
        "incident_id": row.incident_id,
        "author_user_id": row.author_user_id,
        "body": row.body,
        "internal": row.internal,
        "created_at": row.created_at,
    }


@router.get("/services")
def services(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return {"items": svc.list_services(db)}


@router.get("/health-checks")
def health_checks(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return {"items": svc.list_services(db)}


@router.get("/infrastructure", response_model=InfrastructureResponse)
def infrastructure(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return svc.infrastructure(db)


@router.get("/performance", response_model=PerformanceResponse)
def performance(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    hours: int = Query(default=24, ge=1, le=720),
    service: str | None = None,
) -> Any:
    return svc.performance(db, hours=hours, service=service)


@router.get("/ai-tutor", response_model=AiTutorMetricsResponse)
def ai_tutor(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    hours: int = Query(default=24, ge=1, le=720),
) -> Any:
    return svc.ai_tutor_metrics(db, hours=hours)


@router.get("/llm-routing")
def llm_routing(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    """Which provider/model/API key is primary, secondary, next, and last used."""
    return svc.llm_routing(db)


@router.get("/active-users")
def active_users(
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    minutes: int = Query(default=10, ge=1, le=120),
) -> Any:
    """Active users per Ask Learn Mastro / Ask Questions / Practice / Explain / Text / Voice."""
    return svc.active_users(minutes=minutes)


@router.get("/ux", response_model=UxMetricsResponse)
def ux(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    hours: int = Query(default=24, ge=1, le=720),
) -> Any:
    return svc.ux_metrics(db, hours=hours)


@router.get("/alerts", response_model=AlertListResponse)
def alerts(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    status_filter: str | None = Query(default=None, alias="status"),
) -> Any:
    items, total = svc.list_alerts(db, status=status_filter)
    return {"items": [AlertOut.model_validate(i) for i in items], "total": total}


@router.post("/alerts/{alert_id}/ack", response_model=AlertOut)
def alert_ack(
    alert_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    note: str | None = None,
) -> Any:
    row = svc.ack_alert(db, alert_id, int(user.id), note)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found")
    return AlertOut.model_validate(row)


@router.get("/alert-rules", response_model=list[AlertRuleOut])
def alert_rules(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return [AlertRuleOut.model_validate(r) for r in svc.list_alert_rules(db)]


@router.post("/alert-rules", response_model=AlertRuleOut)
def alert_rule_create(
    body: AlertRuleIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return AlertRuleOut.model_validate(svc.create_alert_rule(db, body, int(user.id)))


@router.get("/deployments", response_model=list[DeploymentOut])
def deployments(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    rows = svc.list_deployments(db)
    return [
        DeploymentOut(
            id=r.id,
            service_key=r.service_key,
            environment=r.environment,
            version=r.version,
            commit_sha=r.commit_sha,
            coolify_deployment_id=r.coolify_deployment_id,
            status=r.status,
            started_at=r.started_at,
            finished_at=r.finished_at,
            meta=r.meta,
        )
        for r in rows
    ]


@router.get("/reports", response_model=ReportsResponse)
def reports(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    hours: int = Query(default=168, ge=1, le=2160),
) -> Any:
    return svc.reports(db, hours=hours)


@router.get("/settings", response_model=SettingsOut)
def settings_get(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return svc.get_settings_out(db)


@router.patch("/settings", response_model=SettingsOut)
def settings_patch(
    body: SettingsPatch,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    return svc.patch_settings(db, body, int(user.id))


@router.post("/ingest/events", response_model=IssueOut)
def ingest_events(
    body: IngestEventIn,
    db: Annotated[Session, Depends(get_db)],
    x_monitoring_token: Annotated[str | None, Header(alias="X-Monitoring-Token")] = None,
) -> Any:
    if not _internal_token_ok(x_monitoring_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid ingest token")
    return IssueOut.model_validate(svc.ingest_event(db, body))


@router.post("/ingest/voice")
def ingest_voice(
    body: VoiceIngestIn,
    db: Annotated[Session, Depends(get_db)],
    x_monitoring_token: Annotated[str | None, Header(alias="X-Monitoring-Token")] = None,
    x_voice_token: Annotated[str | None, Header(alias="X-Voice-Token")] = None,
) -> Any:
    token = x_monitoring_token or x_voice_token
    if not _internal_token_ok(token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid ingest token")
    return svc.ingest_voice(db, body)


@router.post("/webhooks/sentry", response_model=IssueOut)
def sentry_webhook(
    body: dict[str, Any],
    db: Annotated[Session, Depends(get_db)],
    x_sentry_secret: Annotated[str | None, Header(alias="X-Sentry-Secret")] = None,
) -> Any:
    expected = os.environ.get("SENTRY_WEBHOOK_SECRET") or ""
    if expected and x_sentry_secret != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid webhook secret")
    event = body.get("event") or body.get("data", {}).get("event") or body
    title = str(event.get("title") or event.get("message") or "Sentry event")[:500]
    ingest = IngestEventIn(
        source="sentry",
        service=str((event.get("tags") or {}).get("service") or "api"),
        environment=str(event.get("environment") or "production"),
        severity=str(event.get("level") or "error"),
        title=title,
        message=str(event.get("message") or title),
        fingerprint=None,
        stacktrace=event.get("exception") if isinstance(event.get("exception"), dict) else None,
        breadcrumbs=event.get("breadcrumbs") if isinstance(event.get("breadcrumbs"), list) else None,
        request=event.get("request") if isinstance(event.get("request"), dict) else None,
        user_ref=str((event.get("user") or {}).get("id") or "") or None,
        release=str(event.get("release") or "") or None,
        meta={"sentry_id": event.get("event_id") or body.get("id")},
    )
    issue = svc.ingest_event(db, ingest)
    if event.get("event_id"):
        issue.sentry_id = str(event.get("event_id"))[:64]
        db.commit()
        db.refresh(issue)
    return IssueOut.model_validate(issue)


@router.post("/ingest/web-vitals")
def ingest_web_vitals(
    body: dict[str, Any],
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
) -> Any:
    """Accept Web Vitals samples from the admin/frontend instrumentation."""
    from datetime import UTC, datetime

    name = str(body.get("name") or body.get("metric") or "").strip()
    value = body.get("value")
    if not name or value is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name and value required")
    try:
        svc._bump_metric(
            db,
            datetime.now(UTC).replace(second=0, microsecond=0),
            f"web_vitals.{name}",
            float(value),
            service_key="frontend",
            environment=os.environ.get("ENVIRONMENT", "production"),
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to store vitals") from exc
    return {"ok": True}

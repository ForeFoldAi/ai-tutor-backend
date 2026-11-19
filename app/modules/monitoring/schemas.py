"""Pydantic schemas for monitoring APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TimeRangeQuery(BaseModel):
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    environment: str | None = "production"
    service: str | None = None


class OverviewResponse(BaseModel):
    overall_status: str
    api_success_rate: float | None = None
    error_rate: float | None = None
    request_volume: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    active_incidents: int = 0
    open_issues: int = 0
    affected_users: int = 0
    cpu_percent: float | None = None
    ram_percent: float | None = None
    disk_percent: float | None = None
    database_status: str = "unknown"
    redis_status: str = "unknown"
    containers: list[dict[str, Any]] = Field(default_factory=list)
    ai_tutor_status: str = "unknown"
    recent_deployments: list[dict[str, Any]] = Field(default_factory=list)
    recent_alerts: list[dict[str, Any]] = Field(default_factory=list)
    services: list[dict[str, Any]] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class IssueOut(BaseModel):
    id: int
    fingerprint: str
    title: str
    culprit: str | None = None
    severity: str
    status: str
    environment: str
    service_key: str | None = None
    release: str | None = None
    occurrence_count: int
    affected_users: int
    assignee_user_id: int | None = None
    first_seen_at: datetime
    last_seen_at: datetime
    resolved_at: datetime | None = None
    meta: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class IssueListResponse(BaseModel):
    items: list[IssueOut]
    total: int
    page: int
    page_size: int


class IssuePatch(BaseModel):
    status: str | None = None
    severity: str | None = None
    assignee_user_id: int | None = None


class IssueCommentIn(BaseModel):
    body: str = Field(min_length=1, max_length=5000)


class IssueCommentOut(BaseModel):
    id: int
    issue_id: int
    author_user_id: int
    body: str
    created_at: datetime

    model_config = {"from_attributes": True}


class IssueDetailResponse(BaseModel):
    issue: IssueOut
    events: list[dict[str, Any]] = Field(default_factory=list)
    comments: list[IssueCommentOut] = Field(default_factory=list)


class LogEntryOut(BaseModel):
    ts: datetime | None = None
    environment: str | None = None
    service: str | None = None
    module: str | None = None
    severity: str | None = None
    message: str = ""
    request_id: str | None = None
    trace_id: str | None = None
    release: str | None = None
    duration_ms: float | None = None
    raw: dict[str, Any] | None = None


class LogListResponse(BaseModel):
    items: list[LogEntryOut]
    next_cursor: str | None = None
    source: str = "loki"
    unavailable: bool = False
    message: str | None = None


class IncidentOut(BaseModel):
    id: int
    title: str
    severity: str
    status: str
    owner_user_id: int | None = None
    environment: str
    service_key: str | None = None
    related_issue_ids: list[Any] | None = None
    root_cause: str | None = None
    postmortem: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    opened_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IncidentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    severity: str = "sev2"
    service_key: str | None = None
    environment: str = "production"
    related_issue_ids: list[int] | None = None


class IncidentPatch(BaseModel):
    status: str | None = None
    severity: str | None = None
    owner_user_id: int | None = None
    root_cause: str | None = None
    postmortem: str | None = None


class IncidentCommentIn(BaseModel):
    body: str = Field(min_length=1, max_length=5000)
    internal: bool = True


class IncidentListResponse(BaseModel):
    items: list[IncidentOut]
    total: int


class IncidentDetailResponse(BaseModel):
    incident: IncidentOut
    events: list[dict[str, Any]] = Field(default_factory=list)
    comments: list[dict[str, Any]] = Field(default_factory=list)


class AlertRuleIn(BaseModel):
    name: str
    metric_key: str
    comparator: str = "gt"
    threshold: float
    window_seconds: int = 300
    consecutive_failures: int = 1
    severity: str = "warning"
    cooldown_seconds: int = 900
    channels: list[str] | None = None
    service_key: str | None = None
    environment: str = "production"
    enabled: bool = True
    detection_mode: str = "threshold"


class AlertRuleOut(AlertRuleIn):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AlertOut(BaseModel):
    id: int
    rule_id: int | None = None
    fingerprint: str
    title: str
    severity: str
    status: str
    service_key: str | None = None
    environment: str
    value: float | None = None
    threshold: float | None = None
    message: str
    incident_id: int | None = None
    fired_at: datetime
    resolved_at: datetime | None = None

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    items: list[AlertOut]
    total: int


class DeploymentOut(BaseModel):
    id: int
    service_key: str
    environment: str
    version: str | None = None
    commit_sha: str | None = None
    coolify_deployment_id: str | None = None
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    meta: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class SettingsOut(BaseModel):
    retention_days_logs: int
    retention_days_metrics: int
    retention_days_events: int
    auto_refresh_seconds: int
    telegram_chat_id: str | None = None
    slack_webhook_configured: bool = False
    alert_email_to: str | None = None
    enabled: bool = True
    meta: dict[str, Any] | None = None


class SettingsPatch(BaseModel):
    retention_days_logs: int | None = None
    retention_days_metrics: int | None = None
    retention_days_events: int | None = None
    auto_refresh_seconds: int | None = None
    telegram_chat_id: str | None = None
    slack_webhook_url: str | None = None
    alert_email_to: str | None = None
    enabled: bool | None = None


class IngestEventIn(BaseModel):
    source: str = "internal"
    service: str = "api"
    environment: str = "production"
    severity: str = "error"
    title: str
    message: str | None = None
    fingerprint: str | None = None
    stacktrace: dict[str, Any] | None = None
    breadcrumbs: list[Any] | None = None
    request: dict[str, Any] | None = None
    user_ref: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    release: str | None = None
    duration_ms: float | None = None
    meta: dict[str, Any] | None = None


class VoiceIngestIn(BaseModel):
    event: str
    session_id: str | None = None
    student_id: str | None = None
    stt_ms: float | None = None
    tts_ms: float | None = None
    rag_ms: float | None = None
    ttfa_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict[str, Any] | None = None


class PerformanceResponse(BaseModel):
    series: list[dict[str, Any]] = Field(default_factory=list)
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    slow_requests: list[dict[str, Any]] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class AiTutorMetricsResponse(BaseModel):
    request_volume: int = 0
    success_rate: float | None = None
    latency_p95_ms: float | None = None
    timeouts: int = 0
    token_prompt: int = 0
    token_completion: int = 0
    model_errors: int = 0
    rate_limits: int = 0
    rag_failures: int = 0
    voice: dict[str, Any] = Field(default_factory=dict)
    vector_status: str = "unknown"
    routing: dict[str, Any] = Field(default_factory=dict)
    active_users: dict[str, Any] = Field(default_factory=dict)
    series: list[dict[str, Any]] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class UxMetricsResponse(BaseModel):
    web_vitals: dict[str, Any] = Field(default_factory=dict)
    series: list[dict[str, Any]] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class InfrastructureResponse(BaseModel):
    cpu_percent: float | None = None
    ram_percent: float | None = None
    disk_percent: float | None = None
    network: dict[str, Any] | None = None
    containers: list[dict[str, Any]] = Field(default_factory=list)
    deployments: list[dict[str, Any]] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    message: str | None = None


class ReportsResponse(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    series: list[dict[str, Any]] = Field(default_factory=list)

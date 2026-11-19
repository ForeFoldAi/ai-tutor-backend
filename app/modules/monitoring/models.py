"""SQLAlchemy models for the `monitoring` PostgreSQL schema."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

_SCHEMA = "monitoring"


class MonService(Base):
    __tablename__ = "services"
    __table_args__ = (
        UniqueConstraint("key", name="uq_mon_services_key"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="api")
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    coolify_uuid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonHealthCheck(Base):
    __tablename__ = "health_checks"
    __table_args__ = (
        Index("ix_mon_health_checks_service_checked", "service_id", "checked_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    service_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    service_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # healthy|degraded|down
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False, index=True
    )


class MonIssue(Base):
    __tablename__ = "issues"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_mon_issues_fingerprint"),
        Index("ix_mon_issues_status_severity", "status", "severity"),
        Index("ix_mon_issues_last_seen", "last_seen_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    culprit: Mapped[str | None] = mapped_column(String(500), nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="error")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="open")
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    service_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    release: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sentry_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    affected_users: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assignee_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonIssueEvent(Base):
    __tablename__ = "issue_events"
    __table_args__ = (
        Index("ix_mon_issue_events_issue_created", "issue_id", "created_at"),
        Index("ix_mon_issue_events_request", "request_id"),
        Index("ix_mon_issue_events_trace", "trace_id"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    issue_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    event_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stacktrace: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    breadcrumbs: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    request: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    user_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    release: Mapped[str | None] = mapped_column(String(128), nullable=True)
    environment: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonIssueComment(Base):
    __tablename__ = "issue_comments"
    __table_args__ = (
        Index("ix_mon_issue_comments_issue", "issue_id", "created_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    issue_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    author_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonLogIndex(Base):
    """Metadata only — raw bodies live in Loki."""

    __tablename__ = "log_index"
    __table_args__ = (
        Index("ix_mon_log_index_request", "request_id"),
        Index("ix_mon_log_index_trace", "trace_id"),
        Index("ix_mon_log_index_ts", "ts"),
        Index("ix_mon_log_index_service_severity", "service", "severity"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    service: Mapped[str] = mapped_column(String(64), nullable=False)
    module: Mapped[str | None] = mapped_column(String(128), nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    message_preview: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    release: Mapped[str | None] = mapped_column(String(128), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    loki_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)


class MonIncident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_mon_incidents_status_severity", "status", "severity"),
        Index("ix_mon_incidents_opened", "opened_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="sev2")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="open")
    owner_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    service_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    related_issue_ids: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    postmortem: Mapped[str | None] = mapped_column(Text, nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonIncidentEvent(Base):
    __tablename__ = "incident_events"
    __table_args__ = (
        Index("ix_mon_incident_events_incident", "incident_id", "created_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    incident_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actor_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonIncidentComment(Base):
    __tablename__ = "incident_comments"
    __table_args__ = (
        Index("ix_mon_incident_comments_incident", "incident_id", "created_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    incident_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    author_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    internal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonAlertRule(Base):
    __tablename__ = "alert_rules"
    __table_args__ = (
        Index("ix_mon_alert_rules_enabled", "enabled"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    metric_key: Mapped[str] = mapped_column(String(128), nullable=False)
    comparator: Mapped[str] = mapped_column(String(8), nullable=False, default="gt")
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warning")
    cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=900)
    channels: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    service_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    detection_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="threshold")
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonAlert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_mon_alerts_status_created", "status", "created_at"),
        Index("ix_mon_alerts_fingerprint", "fingerprint"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rule_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warning")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="firing")
    service_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    incident_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    fired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonAlertAck(Base):
    __tablename__ = "alert_acknowledgements"
    __table_args__ = ({"schema": _SCHEMA},)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    alert_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonDeployment(Base):
    __tablename__ = "deployments"
    __table_args__ = (
        Index("ix_mon_deployments_started", "started_at"),
        Index("ix_mon_deployments_service", "service_key"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    service_key: Mapped[str] = mapped_column(String(64), nullable=False)
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    coolify_deployment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class MonMetricAggregate(Base):
    __tablename__ = "metric_aggregates"
    __table_args__ = (
        UniqueConstraint(
            "bucket_start",
            "metric_key",
            "service_key",
            "environment",
            name="uq_mon_metric_agg_bucket",
        ),
        Index("ix_mon_metric_agg_key_bucket", "metric_key", "bucket_start"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bucket_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    metric_key: Mapped[str] = mapped_column(String(128), nullable=False)
    service_key: Mapped[str] = mapped_column(String(64), nullable=False, default="api")
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    min_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    p99: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class MonAuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_mon_audit_events_created", "created_at"),
        {"schema": _SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class MonSettings(Base):
    __tablename__ = "settings"
    __table_args__ = ({"schema": _SCHEMA},)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    retention_days_logs: Mapped[int] = mapped_column(Integer, nullable=False, default=14)
    retention_days_metrics: Mapped[int] = mapped_column(Integer, nullable=False, default=90)
    retention_days_events: Mapped[int] = mapped_column(Integer, nullable=False, default=90)
    auto_refresh_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    telegram_chat_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    slack_webhook_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    alert_email_to: Mapped[str | None] = mapped_column(String(256), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

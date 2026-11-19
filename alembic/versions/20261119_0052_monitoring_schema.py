"""Create monitoring schema and tables for production observability.

Revision ID: 20261119_0052
Revises: 20261107_0051
Create Date: 2026-11-19

Rollback: alembic downgrade 20261107_0051  (drops monitoring schema only)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261119_0052"
down_revision = "20261107_0051"
branch_labels = None
depends_on = None

SCHEMA = "monitoring"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))

    op.create_table(
        "services",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False, server_default="api"),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("base_url", sa.String(512), nullable=True),
        sa.Column("coolify_uuid", sa.String(64), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("key", name="uq_mon_services_key"),
        schema=SCHEMA,
    )

    op.create_table(
        "health_checks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("service_id", sa.BigInteger(), nullable=True),
        sa.Column("service_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_health_checks_service_checked", "health_checks", ["service_id", "checked_at"], schema=SCHEMA)
    op.create_index("ix_mon_health_checks_service_key", "health_checks", ["service_key"], schema=SCHEMA)
    op.create_index("ix_mon_health_checks_checked_at", "health_checks", ["checked_at"], schema=SCHEMA)

    op.create_table(
        "issues",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("culprit", sa.String(500), nullable=True),
        sa.Column("severity", sa.String(16), nullable=False, server_default="error"),
        sa.Column("status", sa.String(24), nullable=False, server_default="open"),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("service_key", sa.String(64), nullable=True),
        sa.Column("release", sa.String(128), nullable=True),
        sa.Column("sentry_id", sa.String(64), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("affected_users", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("assignee_user_id", sa.BigInteger(), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("fingerprint", name="uq_mon_issues_fingerprint"),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_issues_status_severity", "issues", ["status", "severity"], schema=SCHEMA)
    op.create_index("ix_mon_issues_last_seen", "issues", ["last_seen_at"], schema=SCHEMA)
    op.create_index("ix_mon_issues_service_key", "issues", ["service_key"], schema=SCHEMA)
    op.create_index("ix_mon_issues_sentry_id", "issues", ["sentry_id"], schema=SCHEMA)

    op.create_table(
        "issue_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("issue_id", sa.BigInteger(), nullable=False),
        sa.Column("event_id", sa.String(64), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("stacktrace", postgresql.JSONB(), nullable=True),
        sa.Column("breadcrumbs", postgresql.JSONB(), nullable=True),
        sa.Column("request", postgresql.JSONB(), nullable=True),
        sa.Column("user_ref", sa.String(128), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("release", sa.String(128), nullable=True),
        sa.Column("environment", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_issue_events_issue_created", "issue_events", ["issue_id", "created_at"], schema=SCHEMA)
    op.create_index("ix_mon_issue_events_request", "issue_events", ["request_id"], schema=SCHEMA)
    op.create_index("ix_mon_issue_events_trace", "issue_events", ["trace_id"], schema=SCHEMA)
    op.create_index("ix_mon_issue_events_issue_id", "issue_events", ["issue_id"], schema=SCHEMA)

    op.create_table(
        "issue_comments",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("issue_id", sa.BigInteger(), nullable=False),
        sa.Column("author_user_id", sa.BigInteger(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_issue_comments_issue", "issue_comments", ["issue_id", "created_at"], schema=SCHEMA)

    op.create_table(
        "log_index",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("service", sa.String(64), nullable=False),
        sa.Column("module", sa.String(128), nullable=True),
        sa.Column("severity", sa.String(16), nullable=False, server_default="info"),
        sa.Column("message_preview", sa.String(500), nullable=False, server_default=""),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("release", sa.String(128), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("loki_ref", sa.String(256), nullable=True),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_log_index_request", "log_index", ["request_id"], schema=SCHEMA)
    op.create_index("ix_mon_log_index_trace", "log_index", ["trace_id"], schema=SCHEMA)
    op.create_index("ix_mon_log_index_ts", "log_index", ["ts"], schema=SCHEMA)
    op.create_index("ix_mon_log_index_service_severity", "log_index", ["service", "severity"], schema=SCHEMA)

    op.create_table(
        "incidents",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False, server_default="sev2"),
        sa.Column("status", sa.String(24), nullable=False, server_default="open"),
        sa.Column("owner_user_id", sa.BigInteger(), nullable=True),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("service_key", sa.String(64), nullable=True),
        sa.Column("related_issue_ids", postgresql.JSONB(), nullable=True),
        sa.Column("root_cause", sa.Text(), nullable=True),
        sa.Column("postmortem", sa.Text(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_incidents_status_severity", "incidents", ["status", "severity"], schema=SCHEMA)
    op.create_index("ix_mon_incidents_opened", "incidents", ["opened_at"], schema=SCHEMA)

    op.create_table(
        "incident_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("incident_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(48), nullable=False),
        sa.Column("message", sa.Text(), nullable=False, server_default=""),
        sa.Column("actor_user_id", sa.BigInteger(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_incident_events_incident", "incident_events", ["incident_id", "created_at"], schema=SCHEMA)

    op.create_table(
        "incident_comments",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("incident_id", sa.BigInteger(), nullable=False),
        sa.Column("author_user_id", sa.BigInteger(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("internal", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_mon_incident_comments_incident", "incident_comments", ["incident_id", "created_at"], schema=SCHEMA
    )

    op.create_table(
        "alert_rules",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("metric_key", sa.String(128), nullable=False),
        sa.Column("comparator", sa.String(8), nullable=False, server_default="gt"),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("window_seconds", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warning"),
        sa.Column("cooldown_seconds", sa.Integer(), nullable=False, server_default="900"),
        sa.Column("channels", postgresql.JSONB(), nullable=True),
        sa.Column("service_key", sa.String(64), nullable=True),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("detection_mode", sa.String(32), nullable=False, server_default="threshold"),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_alert_rules_enabled", "alert_rules", ["enabled"], schema=SCHEMA)

    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("rule_id", sa.BigInteger(), nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warning"),
        sa.Column("status", sa.String(24), nullable=False, server_default="firing"),
        sa.Column("service_key", sa.String(64), nullable=True),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False, server_default=""),
        sa.Column("incident_id", sa.BigInteger(), nullable=True),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_notified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_alerts_status_created", "alerts", ["status", "created_at"], schema=SCHEMA)
    op.create_index("ix_mon_alerts_fingerprint", "alerts", ["fingerprint"], schema=SCHEMA)
    op.create_index("ix_mon_alerts_rule_id", "alerts", ["rule_id"], schema=SCHEMA)

    op.create_table(
        "alert_acknowledgements",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("alert_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_alert_acks_alert_id", "alert_acknowledgements", ["alert_id"], schema=SCHEMA)

    op.create_table(
        "deployments",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("service_key", sa.String(64), nullable=False),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("version", sa.String(128), nullable=True),
        sa.Column("commit_sha", sa.String(64), nullable=True),
        sa.Column("coolify_deployment_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_deployments_started", "deployments", ["started_at"], schema=SCHEMA)
    op.create_index("ix_mon_deployments_service", "deployments", ["service_key"], schema=SCHEMA)
    op.create_index("ix_mon_deployments_coolify", "deployments", ["coolify_deployment_id"], schema=SCHEMA)

    op.create_table(
        "metric_aggregates",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("bucket_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bucket_seconds", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("metric_key", sa.String(128), nullable=False),
        sa.Column("service_key", sa.String(64), nullable=False, server_default="api"),
        sa.Column("environment", sa.String(32), nullable=False, server_default="production"),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sum_value", sa.Float(), nullable=False, server_default="0"),
        sa.Column("min_value", sa.Float(), nullable=True),
        sa.Column("max_value", sa.Float(), nullable=True),
        sa.Column("p50", sa.Float(), nullable=True),
        sa.Column("p95", sa.Float(), nullable=True),
        sa.Column("p99", sa.Float(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.UniqueConstraint(
            "bucket_start", "metric_key", "service_key", "environment", name="uq_mon_metric_agg_bucket"
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_metric_agg_key_bucket", "metric_aggregates", ["metric_key", "bucket_start"], schema=SCHEMA)

    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("actor_user_id", sa.BigInteger(), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=True),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )
    op.create_index("ix_mon_audit_events_created", "audit_events", ["created_at"], schema=SCHEMA)

    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("retention_days_logs", sa.Integer(), nullable=False, server_default="14"),
        sa.Column("retention_days_metrics", sa.Integer(), nullable=False, server_default="90"),
        sa.Column("retention_days_events", sa.Integer(), nullable=False, server_default="90"),
        sa.Column("auto_refresh_seconds", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("telegram_chat_id", sa.String(64), nullable=True),
        sa.Column("slack_webhook_url", sa.String(512), nullable=True),
        sa.Column("alert_email_to", sa.String(256), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        schema=SCHEMA,
    )

    # Seed default services + settings
    op.execute(
        sa.text(
            f"""
            INSERT INTO {SCHEMA}.settings (id) VALUES (1) ON CONFLICT DO NOTHING;
            INSERT INTO {SCHEMA}.services (key, name, kind, environment) VALUES
              ('api', 'FastAPI API', 'api', 'production'),
              ('worker', 'Celery Worker', 'worker', 'production'),
              ('voice', 'Voice RTC', 'voice', 'production'),
              ('frontend', 'Website', 'frontend', 'production'),
              ('postgres', 'PostgreSQL', 'database', 'production'),
              ('redis', 'Redis', 'cache', 'production'),
              ('qdrant', 'Qdrant', 'vector', 'production')
            ON CONFLICT (key) DO NOTHING;
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))

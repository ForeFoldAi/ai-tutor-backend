"""Alembic revision: Learning Intelligence Agent tables.

Revision ID: 20260721_0039
Revises: 20260721_0038
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260721_0039"
down_revision = "20260721_0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lia_learning_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("school_id", sa.BigInteger(), sa.ForeignKey("schools.id", ondelete="SET NULL"), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scope_key", sa.String(512), nullable=False, server_default=""),
        sa.Column("subject_name", sa.String(120), nullable=True),
        sa.Column("chapter_id", sa.BigInteger(), nullable=True),
        sa.Column("chapter_name", sa.String(150), nullable=True),
        sa.Column("concept_key", sa.String(256), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_lia_learning_events_student", "lia_learning_events", ["student_user_id"])
    op.create_index("ix_lia_learning_events_type", "lia_learning_events", ["event_type"])
    op.create_index("ix_lia_learning_events_occurred", "lia_learning_events", ["occurred_at"])
    op.create_index("ix_lia_learning_events_scope", "lia_learning_events", ["scope_key"])
    op.create_index("ix_lia_learning_events_concept", "lia_learning_events", ["concept_key"])

    op.create_table(
        "lia_student_profiles",
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("learning_style", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("teaching_preferences", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("memory_profile", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("strong_subjects", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("weak_subjects", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("strong_concepts", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("weak_concepts", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("attention_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("engagement_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("motivation_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("persistence_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("improvement_trend", sa.Float(), nullable=False, server_default="0"),
        sa.Column("regression_trend", sa.Float(), nullable=False, server_default="0"),
        sa.Column("risk_level", sa.String(16), nullable=False, server_default="low"),
        sa.Column("revision_behaviour", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("twin_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "lia_concepts",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("concept_key", sa.String(256), nullable=False),
        sa.Column("display_name", sa.String(256), nullable=False),
        sa.Column("subject_name", sa.String(120), nullable=True),
        sa.Column("chapter_name", sa.String(150), nullable=True),
        sa.Column("bloom_default", sa.String(32), nullable=False, server_default="understand"),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.UniqueConstraint("concept_key", name="uq_lia_concepts_key"),
    )
    op.create_index("ix_lia_concepts_key", "lia_concepts", ["concept_key"])
    op.create_index("ix_lia_concepts_subject", "lia_concepts", ["subject_name"])

    op.create_table(
        "lia_concept_dependencies",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("prerequisite_key", sa.String(256), nullable=False),
        sa.Column("dependent_key", sa.String(256), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1"),
        sa.UniqueConstraint("prerequisite_key", "dependent_key", name="uq_lia_concept_dep"),
    )
    op.create_index("ix_lia_concept_dep_prereq", "lia_concept_dependencies", ["prerequisite_key"])
    op.create_index("ix_lia_concept_dep_dependent", "lia_concept_dependencies", ["dependent_key"])

    op.create_table(
        "lia_concept_mastery",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("concept_key", sa.String(256), nullable=False),
        sa.Column("mastery_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("understanding_level", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("memorized_likelihood", sa.Float(), nullable=False, server_default="0"),
        sa.Column("evidence_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("student_user_id", "concept_key", name="uq_lia_concept_mastery"),
    )
    op.create_index("ix_lia_concept_mastery_student", "lia_concept_mastery", ["student_user_id"])
    op.create_index("ix_lia_concept_mastery_concept", "lia_concept_mastery", ["concept_key"])

    op.create_table(
        "lia_misconceptions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("concept_key", sa.String(256), nullable=False),
        sa.Column("misconception_key", sa.String(256), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("evidence", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("student_user_id", "concept_key", "misconception_key", name="uq_lia_misconception"),
    )
    op.create_index("ix_lia_misconceptions_student", "lia_misconceptions", ["student_user_id"])

    op.create_table(
        "lia_knowledge_gaps",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("concept_key", sa.String(256), nullable=False),
        sa.Column("gap_type", sa.String(64), nullable=False, server_default="incomplete"),
        sa.Column("missing_prerequisites", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("student_user_id", "concept_key", name="uq_lia_knowledge_gap"),
    )
    op.create_index("ix_lia_knowledge_gaps_student", "lia_knowledge_gaps", ["student_user_id"])

    op.create_table(
        "lia_metric_history",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("metric_type", sa.String(32), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("context", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lia_metric_history_student", "lia_metric_history", ["student_user_id"])
    op.create_index("ix_lia_metric_history_type", "lia_metric_history", ["metric_type"])
    op.create_index("ix_lia_metric_history_recorded", "lia_metric_history", ["recorded_at"])

    op.create_table(
        "lia_teaching_memory",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("concept_key", sa.String(256), nullable=False),
        sa.Column("method_key", sa.String(128), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_outcome", sa.String(16), nullable=False, server_default="neutral"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("student_user_id", "concept_key", "method_key", name="uq_lia_teaching_memory"),
    )
    op.create_index("ix_lia_teaching_memory_student", "lia_teaching_memory", ["student_user_id"])

    op.create_table(
        "lia_tutor_guidance_cache",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scope_key", sa.String(512), nullable=False),
        sa.Column("topic_hash", sa.String(64), nullable=False),
        sa.Column("guidance_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lia_guidance_cache_lookup", "lia_tutor_guidance_cache", ["student_user_id", "scope_key", "topic_hash"])

    op.create_table(
        "lia_teacher_recommendations",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tutor_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("summary_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lia_teacher_rec_student", "lia_teacher_recommendations", ["student_user_id"])

    op.create_table(
        "lia_period_summaries",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("period_type", sa.String(16), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("summary_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lia_period_summaries_student", "lia_period_summaries", ["student_user_id", "period_type"])

    op.create_table(
        "lia_risk_assessments",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("factors", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lia_risk_student", "lia_risk_assessments", ["student_user_id"])

    op.create_table(
        "lia_learning_predictions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("student_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("concept_key", sa.String(256), nullable=True),
        sa.Column("prediction_type", sa.String(64), nullable=False),
        sa.Column("prediction_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lia_predictions_student", "lia_learning_predictions", ["student_user_id"])


def downgrade() -> None:
    for table in (
        "lia_learning_predictions",
        "lia_risk_assessments",
        "lia_period_summaries",
        "lia_teacher_recommendations",
        "lia_tutor_guidance_cache",
        "lia_teaching_memory",
        "lia_metric_history",
        "lia_knowledge_gaps",
        "lia_misconceptions",
        "lia_concept_mastery",
        "lia_concept_dependencies",
        "lia_concepts",
        "lia_student_profiles",
        "lia_learning_events",
    ):
        op.drop_table(table)

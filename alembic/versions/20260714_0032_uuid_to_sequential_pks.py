"""Convert UUID primary/foreign keys to BIGINT sequential IDs.

Revision ID: 20260714_0032
Revises: 20260714_0031
Create Date: 2026-07-14

PostgreSQL only. Preserves public sequential values:
  - users.id    <- users.account_id  (then drop account_id)
  - schools.id  <- schools.seq       (then drop seq)

school_subjects / school_classes keep per-school `seq` (API); new global bigint PK via bigserial.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0032"
down_revision = "20260714_0031"
branch_labels = None
depends_on = None

# Tables whose UUID PKs / FKs are rewritten in this revision.
_AFFECTED_TABLES = (
    "users",
    "user_settings",
    "user_signup_profiles",
    "sessions",
    "schools",
    "school_subjects",
    "school_classes",
    "school_class_subjects",
    "board_definitions",
    "syllabus_subjects",
    "textbook_uploads",
    "textbook_images",
    "lesson_plans",
    "lesson_plan_artifacts",
    "lesson_plan_jobs",
    "lesson_plan_versions",
    "lesson_plan_exports",
)

# UUID-PK tables that get a fresh bigserial id_new (not users/schools).
_BIGSERIAL_PK_TABLES = (
    "sessions",
    "school_subjects",
    "school_classes",
    "school_class_subjects",
    "board_definitions",
    "syllabus_subjects",
    "textbook_uploads",
    "textbook_images",
    "lesson_plans",
    "lesson_plan_jobs",
    "lesson_plan_artifacts",
    "lesson_plan_versions",
    "lesson_plan_exports",
)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _execute(sql: str) -> None:
    op.execute(sa.text(sql))


def _drop_fks_involving(tables: tuple[str, ...]) -> None:
    """Drop every FK that references or is defined on any of `tables`."""
    table_list = ", ".join(f"'{t}'" for t in tables)
    _execute(
        f"""
        DO $$
        DECLARE
            r RECORD;
        BEGIN
            FOR r IN
                SELECT c.conname AS conname, rel.relname AS table_name
                FROM pg_constraint c
                JOIN pg_class rel ON rel.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = rel.relnamespace
                LEFT JOIN pg_class ref ON ref.oid = c.confrelid
                WHERE c.contype = 'f'
                  AND n.nspname = 'public'
                  AND (
                      rel.relname IN ({table_list})
                      OR ref.relname IN ({table_list})
                  )
            LOOP
                EXECUTE format(
                    'ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I',
                    r.table_name,
                    r.conname
                );
            END LOOP;
        END $$;
        """
    )


def _drop_constraint_if_exists(table: str, name: str) -> None:
    _execute(f'ALTER TABLE {_q(table)} DROP CONSTRAINT IF EXISTS {_q(name)}')


def _drop_index_if_exists(name: str) -> None:
    _execute(f"DROP INDEX IF EXISTS {_q(name)}")


def _attach_id_sequence(table: str, column: str = "id") -> None:
    """Own a sequence by column, setval safely for empty + non-empty tables."""
    seq = f"{table}_{column}_seq"
    t, c = _q(table), _q(column)
    # Sequence name as a SQL string literal for setval/nextval (not an identifier).
    seq_lit = "'" + seq.replace("'", "''") + "'"
    _execute(f"CREATE SEQUENCE IF NOT EXISTS {_q(seq)}")
    _execute(
        f"""
        SELECT setval(
            {seq_lit}::regclass,
            COALESCE((SELECT MAX({c}) FROM {t}), 1),
            (SELECT COUNT(*) > 0 FROM {t})
        )
        """
    )
    _execute(f"ALTER TABLE {t} ALTER COLUMN {c} SET DEFAULT nextval({seq_lit}::regclass)")
    _execute(f"ALTER SEQUENCE {_q(seq)} OWNED BY {t}.{c}")


def _rename_bigserial_sequence(table: str) -> None:
    """After id_new → id, rename {table}_id_new_seq → {table}_id_seq if present."""
    old_seq = f"{table}_id_new_seq"
    new_seq = f"{table}_id_seq"
    _execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_class WHERE relname = '{old_seq}' AND relkind = 'S') THEN
                IF EXISTS (SELECT 1 FROM pg_class WHERE relname = '{new_seq}' AND relkind = 'S') THEN
                    EXECUTE 'DROP SEQUENCE {_q(new_seq)}';
                END IF;
                EXECUTE 'ALTER SEQUENCE {_q(old_seq)} RENAME TO {_q(new_seq)}';
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # 1. Drop ALL foreign keys on/against affected tables (recreated at end).
    # -------------------------------------------------------------------------
    _drop_fks_involving(_AFFECTED_TABLES)

    # Unique / PK constraints that include UUID FK columns must go before column drops.
    for table, name in (
        ("school_subjects", "uq_school_subjects_school_code"),
        ("school_subjects", "uq_school_subjects_school_seq"),
        ("school_classes", "uq_school_classes_school_grade_section_curriculum"),
        ("school_classes", "uq_school_classes_school_seq"),
        ("school_class_subjects", "uq_school_class_subjects"),
        ("lesson_plan_artifacts", "uq_lesson_artifact_version"),
        ("schools", "uq_schools_seq"),
        ("user_settings", "user_settings_pkey"),
        ("user_signup_profiles", "user_signup_profiles_pkey"),
    ):
        _drop_constraint_if_exists(table, name)

    # Drop primary keys on UUID-id tables (constraint name = {table}_pkey).
    for table in (
        "users",
        "schools",
        *_BIGSERIAL_PK_TABLES,
    ):
        _drop_constraint_if_exists(table, f"{table}_pkey")

    # Indexes that will be invalid after column type/name changes (IF EXISTS).
    for idx in (
        "ix_users_account_id",
        "ix_sessions_user_id",
        "ix_school_subjects_school_id",
        "ix_school_classes_school_id",
        "ix_school_class_subjects_school_class_id",
        "ix_school_class_subjects_subject_id",
        "ix_textbook_images_textbook_upload_id",
        "ix_textbook_images_upload_page",
        "ix_lesson_plans_user_id",
        "ix_lesson_plans_user_id_deleted_at",
        "ix_lesson_plan_jobs_lesson_plan_id",
        "ix_lesson_plan_jobs_user_id",
        "ix_lesson_plan_artifacts_lesson_plan_id",
        "ix_lesson_plan_artifacts_job_id",
        "ix_lesson_plan_versions_lesson_plan_id",
        "ix_lesson_plan_versions_plan_version",
        "ix_lesson_plan_exports_lesson_plan_id",
    ):
        _drop_index_if_exists(idx)

    # -------------------------------------------------------------------------
    # 2. Add id_new to every UUID-PK table.
    #    users/schools: BIGINT filled from account_id/seq (preserve public ids).
    #    others: BIGSERIAL (existing rows auto-numbered).
    # -------------------------------------------------------------------------
    _execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS id_new BIGINT")
    _execute("UPDATE users SET id_new = account_id WHERE id_new IS NULL")
    _execute("ALTER TABLE users ALTER COLUMN id_new SET NOT NULL")

    _execute("ALTER TABLE schools ADD COLUMN IF NOT EXISTS id_new BIGINT")
    _execute("UPDATE schools SET id_new = seq WHERE id_new IS NULL")
    _execute("ALTER TABLE schools ALTER COLUMN id_new SET NOT NULL")

    for table in _BIGSERIAL_PK_TABLES:
        # BIGSERIAL adds sequence + fills existing rows; empty tables stay empty.
        _execute(f"ALTER TABLE {_q(table)} ADD COLUMN IF NOT EXISTS id_new BIGSERIAL")

    # -------------------------------------------------------------------------
    # 3. Add *_new FK columns and remap via JOIN on still-present UUID ids.
    # -------------------------------------------------------------------------
    # users → schools / self
    _execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS school_id_new BIGINT")
    _execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_by_new BIGINT")
    _execute(
        """
        UPDATE users u
        SET school_id_new = s.id_new
        FROM schools s
        WHERE u.school_id IS NOT NULL AND u.school_id = s.id
        """
    )
    _execute(
        """
        UPDATE users u
        SET created_by_new = c.id_new
        FROM users c
        WHERE u.created_by IS NOT NULL AND u.created_by = c.id
        """
    )

    # sessions.user_id
    _execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id_new BIGINT")
    _execute(
        """
        UPDATE sessions s
        SET user_id_new = u.id_new
        FROM users u
        WHERE s.user_id = u.id
        """
    )
    _execute("ALTER TABLE sessions ALTER COLUMN user_id_new SET NOT NULL")

    # user_settings / user_signup_profiles: PK is the FK
    _execute("ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS user_id_new BIGINT")
    _execute(
        """
        UPDATE user_settings us
        SET user_id_new = u.id_new
        FROM users u
        WHERE us.user_id = u.id
        """
    )
    _execute("ALTER TABLE user_settings ALTER COLUMN user_id_new SET NOT NULL")

    _execute("ALTER TABLE user_signup_profiles ADD COLUMN IF NOT EXISTS user_id_new BIGINT")
    _execute(
        """
        UPDATE user_signup_profiles usp
        SET user_id_new = u.id_new
        FROM users u
        WHERE usp.user_id = u.id
        """
    )
    _execute("ALTER TABLE user_signup_profiles ALTER COLUMN user_id_new SET NOT NULL")

    # school_subjects / school_classes → schools
    _execute("ALTER TABLE school_subjects ADD COLUMN IF NOT EXISTS school_id_new BIGINT")
    _execute(
        """
        UPDATE school_subjects ss
        SET school_id_new = s.id_new
        FROM schools s
        WHERE ss.school_id = s.id
        """
    )
    _execute("ALTER TABLE school_subjects ALTER COLUMN school_id_new SET NOT NULL")

    _execute("ALTER TABLE school_classes ADD COLUMN IF NOT EXISTS school_id_new BIGINT")
    _execute(
        """
        UPDATE school_classes sc
        SET school_id_new = s.id_new
        FROM schools s
        WHERE sc.school_id = s.id
        """
    )
    _execute("ALTER TABLE school_classes ALTER COLUMN school_id_new SET NOT NULL")

    # school_class_subjects → classes / subjects
    _execute("ALTER TABLE school_class_subjects ADD COLUMN IF NOT EXISTS school_class_id_new BIGINT")
    _execute("ALTER TABLE school_class_subjects ADD COLUMN IF NOT EXISTS subject_id_new BIGINT")
    _execute(
        """
        UPDATE school_class_subjects scs
        SET school_class_id_new = sc.id_new
        FROM school_classes sc
        WHERE scs.school_class_id = sc.id
        """
    )
    _execute(
        """
        UPDATE school_class_subjects scs
        SET subject_id_new = ss.id_new
        FROM school_subjects ss
        WHERE scs.subject_id = ss.id
        """
    )
    _execute("ALTER TABLE school_class_subjects ALTER COLUMN school_class_id_new SET NOT NULL")
    _execute("ALTER TABLE school_class_subjects ALTER COLUMN subject_id_new SET NOT NULL")

    # textbook_uploads.uploaded_by → users (nullable)
    _execute("ALTER TABLE textbook_uploads ADD COLUMN IF NOT EXISTS uploaded_by_new BIGINT")
    _execute(
        """
        UPDATE textbook_uploads t
        SET uploaded_by_new = u.id_new
        FROM users u
        WHERE t.uploaded_by IS NOT NULL AND t.uploaded_by = u.id
        """
    )

    # textbook_images → uploads
    _execute("ALTER TABLE textbook_images ADD COLUMN IF NOT EXISTS textbook_upload_id_new BIGINT")
    _execute(
        """
        UPDATE textbook_images ti
        SET textbook_upload_id_new = tu.id_new
        FROM textbook_uploads tu
        WHERE ti.textbook_upload_id = tu.id
        """
    )
    _execute("ALTER TABLE textbook_images ALTER COLUMN textbook_upload_id_new SET NOT NULL")

    # lesson_plans.user_id
    _execute("ALTER TABLE lesson_plans ADD COLUMN IF NOT EXISTS user_id_new BIGINT")
    _execute(
        """
        UPDATE lesson_plans lp
        SET user_id_new = u.id_new
        FROM users u
        WHERE lp.user_id = u.id
        """
    )
    _execute("ALTER TABLE lesson_plans ALTER COLUMN user_id_new SET NOT NULL")

    # lesson_plan_jobs
    _execute("ALTER TABLE lesson_plan_jobs ADD COLUMN IF NOT EXISTS lesson_plan_id_new BIGINT")
    _execute("ALTER TABLE lesson_plan_jobs ADD COLUMN IF NOT EXISTS user_id_new BIGINT")
    _execute(
        """
        UPDATE lesson_plan_jobs j
        SET lesson_plan_id_new = lp.id_new
        FROM lesson_plans lp
        WHERE j.lesson_plan_id IS NOT NULL AND j.lesson_plan_id = lp.id
        """
    )
    _execute(
        """
        UPDATE lesson_plan_jobs j
        SET user_id_new = u.id_new
        FROM users u
        WHERE j.user_id = u.id
        """
    )
    _execute("ALTER TABLE lesson_plan_jobs ALTER COLUMN user_id_new SET NOT NULL")

    # lesson_plan_artifacts
    _execute("ALTER TABLE lesson_plan_artifacts ADD COLUMN IF NOT EXISTS lesson_plan_id_new BIGINT")
    _execute("ALTER TABLE lesson_plan_artifacts ADD COLUMN IF NOT EXISTS job_id_new BIGINT")
    _execute(
        """
        UPDATE lesson_plan_artifacts a
        SET lesson_plan_id_new = lp.id_new
        FROM lesson_plans lp
        WHERE a.lesson_plan_id = lp.id
        """
    )
    _execute(
        """
        UPDATE lesson_plan_artifacts a
        SET job_id_new = j.id_new
        FROM lesson_plan_jobs j
        WHERE a.job_id IS NOT NULL AND a.job_id = j.id
        """
    )
    _execute("ALTER TABLE lesson_plan_artifacts ALTER COLUMN lesson_plan_id_new SET NOT NULL")

    # lesson_plan_versions
    _execute("ALTER TABLE lesson_plan_versions ADD COLUMN IF NOT EXISTS lesson_plan_id_new BIGINT")
    _execute("ALTER TABLE lesson_plan_versions ADD COLUMN IF NOT EXISTS created_by_new BIGINT")
    _execute(
        """
        UPDATE lesson_plan_versions v
        SET lesson_plan_id_new = lp.id_new
        FROM lesson_plans lp
        WHERE v.lesson_plan_id = lp.id
        """
    )
    _execute(
        """
        UPDATE lesson_plan_versions v
        SET created_by_new = u.id_new
        FROM users u
        WHERE v.created_by IS NOT NULL AND v.created_by = u.id
        """
    )
    _execute("ALTER TABLE lesson_plan_versions ALTER COLUMN lesson_plan_id_new SET NOT NULL")

    # lesson_plan_exports
    _execute("ALTER TABLE lesson_plan_exports ADD COLUMN IF NOT EXISTS lesson_plan_id_new BIGINT")
    _execute("ALTER TABLE lesson_plan_exports ADD COLUMN IF NOT EXISTS version_id_new BIGINT")
    _execute(
        """
        UPDATE lesson_plan_exports e
        SET lesson_plan_id_new = lp.id_new
        FROM lesson_plans lp
        WHERE e.lesson_plan_id = lp.id
        """
    )
    _execute(
        """
        UPDATE lesson_plan_exports e
        SET version_id_new = v.id_new
        FROM lesson_plan_versions v
        WHERE e.version_id IS NOT NULL AND e.version_id = v.id
        """
    )
    _execute("ALTER TABLE lesson_plan_exports ALTER COLUMN lesson_plan_id_new SET NOT NULL")

    # -------------------------------------------------------------------------
    # 4. Drop old UUID columns (PK + FKs) and sequential dual-id columns.
    # -------------------------------------------------------------------------
    # Child FK columns first, then parent UUID PKs.
    _drop_cols = [
        ("users", "school_id"),
        ("users", "created_by"),
        ("users", "id"),
        ("users", "account_id"),
        ("sessions", "user_id"),
        ("sessions", "id"),
        ("user_settings", "user_id"),
        ("user_signup_profiles", "user_id"),
        ("school_subjects", "school_id"),
        ("school_subjects", "id"),
        ("school_classes", "school_id"),
        ("school_classes", "id"),
        ("school_class_subjects", "school_class_id"),
        ("school_class_subjects", "subject_id"),
        ("school_class_subjects", "id"),
        ("schools", "id"),
        ("schools", "seq"),
        ("textbook_uploads", "uploaded_by"),
        ("textbook_uploads", "id"),
        ("textbook_images", "textbook_upload_id"),
        ("textbook_images", "id"),
        ("board_definitions", "id"),
        ("syllabus_subjects", "id"),
        ("lesson_plans", "user_id"),
        ("lesson_plans", "id"),
        ("lesson_plan_jobs", "lesson_plan_id"),
        ("lesson_plan_jobs", "user_id"),
        ("lesson_plan_jobs", "id"),
        ("lesson_plan_artifacts", "lesson_plan_id"),
        ("lesson_plan_artifacts", "job_id"),
        ("lesson_plan_artifacts", "id"),
        ("lesson_plan_versions", "lesson_plan_id"),
        ("lesson_plan_versions", "created_by"),
        ("lesson_plan_versions", "id"),
        ("lesson_plan_exports", "lesson_plan_id"),
        ("lesson_plan_exports", "version_id"),
        ("lesson_plan_exports", "id"),
    ]
    for table, col in _drop_cols:
        _execute(f"ALTER TABLE {_q(table)} DROP COLUMN IF EXISTS {_q(col)} CASCADE")

    # Orphan dual-id sequences (OWNED BY may already drop them with the column).
    _execute("DROP SEQUENCE IF EXISTS users_account_id_seq CASCADE")
    _execute("DROP SEQUENCE IF EXISTS schools_seq_seq CASCADE")

    # -------------------------------------------------------------------------
    # 5. Rename id_new / *_new → final column names.
    # -------------------------------------------------------------------------
    renames = [
        ("users", "id_new", "id"),
        ("users", "school_id_new", "school_id"),
        ("users", "created_by_new", "created_by"),
        ("schools", "id_new", "id"),
        ("sessions", "id_new", "id"),
        ("sessions", "user_id_new", "user_id"),
        ("user_settings", "user_id_new", "user_id"),
        ("user_signup_profiles", "user_id_new", "user_id"),
        ("school_subjects", "id_new", "id"),
        ("school_subjects", "school_id_new", "school_id"),
        ("school_classes", "id_new", "id"),
        ("school_classes", "school_id_new", "school_id"),
        ("school_class_subjects", "id_new", "id"),
        ("school_class_subjects", "school_class_id_new", "school_class_id"),
        ("school_class_subjects", "subject_id_new", "subject_id"),
        ("board_definitions", "id_new", "id"),
        ("syllabus_subjects", "id_new", "id"),
        ("textbook_uploads", "id_new", "id"),
        ("textbook_uploads", "uploaded_by_new", "uploaded_by"),
        ("textbook_images", "id_new", "id"),
        ("textbook_images", "textbook_upload_id_new", "textbook_upload_id"),
        ("lesson_plans", "id_new", "id"),
        ("lesson_plans", "user_id_new", "user_id"),
        ("lesson_plan_jobs", "id_new", "id"),
        ("lesson_plan_jobs", "lesson_plan_id_new", "lesson_plan_id"),
        ("lesson_plan_jobs", "user_id_new", "user_id"),
        ("lesson_plan_artifacts", "id_new", "id"),
        ("lesson_plan_artifacts", "lesson_plan_id_new", "lesson_plan_id"),
        ("lesson_plan_artifacts", "job_id_new", "job_id"),
        ("lesson_plan_versions", "id_new", "id"),
        ("lesson_plan_versions", "lesson_plan_id_new", "lesson_plan_id"),
        ("lesson_plan_versions", "created_by_new", "created_by"),
        ("lesson_plan_exports", "id_new", "id"),
        ("lesson_plan_exports", "lesson_plan_id_new", "lesson_plan_id"),
        ("lesson_plan_exports", "version_id_new", "version_id"),
    ]
    for table, old, new in renames:
        _execute(f"ALTER TABLE {_q(table)} RENAME COLUMN {_q(old)} TO {_q(new)}")

    for table in _BIGSERIAL_PK_TABLES:
        _rename_bigserial_sequence(table)

    # -------------------------------------------------------------------------
    # 6. Recreate primary keys.
    # -------------------------------------------------------------------------
    for table in (
        "users",
        "schools",
        "sessions",
        "school_subjects",
        "school_classes",
        "school_class_subjects",
        "board_definitions",
        "syllabus_subjects",
        "textbook_uploads",
        "textbook_images",
        "lesson_plans",
        "lesson_plan_jobs",
        "lesson_plan_artifacts",
        "lesson_plan_versions",
        "lesson_plan_exports",
    ):
        _execute(f"ALTER TABLE {_q(table)} ADD PRIMARY KEY (id)")

    _execute("ALTER TABLE user_settings ADD PRIMARY KEY (user_id)")
    _execute("ALTER TABLE user_signup_profiles ADD PRIMARY KEY (user_id)")

    # users / schools: attach sequences keyed off preserved max(id) (works for empty).
    _attach_id_sequence("users")
    _attach_id_sequence("schools")

    # -------------------------------------------------------------------------
    # 7. Recreate unique constraints that involve remapped FKs / seq.
    # -------------------------------------------------------------------------
    _execute(
        "ALTER TABLE school_subjects "
        "ADD CONSTRAINT uq_school_subjects_school_code UNIQUE (school_id, code)"
    )
    _execute(
        "ALTER TABLE school_subjects "
        "ADD CONSTRAINT uq_school_subjects_school_seq UNIQUE (school_id, seq)"
    )
    _execute(
        "ALTER TABLE school_classes "
        "ADD CONSTRAINT uq_school_classes_school_grade_section_curriculum "
        "UNIQUE (school_id, grade, section, curriculum)"
    )
    _execute(
        "ALTER TABLE school_classes "
        "ADD CONSTRAINT uq_school_classes_school_seq UNIQUE (school_id, seq)"
    )
    _execute(
        "ALTER TABLE school_class_subjects "
        "ADD CONSTRAINT uq_school_class_subjects UNIQUE (school_class_id, subject_id)"
    )
    _execute(
        "ALTER TABLE lesson_plan_artifacts "
        "ADD CONSTRAINT uq_lesson_artifact_version "
        "UNIQUE (lesson_plan_id, artifact_type, version_number)"
    )

    # -------------------------------------------------------------------------
    # 8. Recreate indexes on remapped FK columns (non-FK indexes untouched).
    # -------------------------------------------------------------------------
    _execute("CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id)")
    # refresh_token_hash unique index should still exist; recreate if CASCADE wiped it
    _execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_sessions_refresh_token_hash "
        "ON sessions (refresh_token_hash)"
    )
    _execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email)")
    _execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_user_settings_username "
        "ON user_settings (username)"
    )

    _execute(
        "CREATE INDEX IF NOT EXISTS ix_school_subjects_school_id "
        "ON school_subjects (school_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_school_classes_school_id "
        "ON school_classes (school_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_school_class_subjects_school_class_id "
        "ON school_class_subjects (school_class_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_school_class_subjects_subject_id "
        "ON school_class_subjects (subject_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_textbook_images_textbook_upload_id "
        "ON textbook_images (textbook_upload_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_textbook_images_upload_page "
        "ON textbook_images (textbook_upload_id, page_index)"
    )

    _execute("CREATE INDEX IF NOT EXISTS ix_lesson_plans_user_id ON lesson_plans (user_id)")
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plans_user_id_deleted_at "
        "ON lesson_plans (user_id, deleted_at)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_jobs_lesson_plan_id "
        "ON lesson_plan_jobs (lesson_plan_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_jobs_user_id "
        "ON lesson_plan_jobs (user_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_artifacts_lesson_plan_id "
        "ON lesson_plan_artifacts (lesson_plan_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_artifacts_job_id "
        "ON lesson_plan_artifacts (job_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_versions_lesson_plan_id "
        "ON lesson_plan_versions (lesson_plan_id)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_versions_plan_version "
        "ON lesson_plan_versions (lesson_plan_id, version_number)"
    )
    _execute(
        "CREATE INDEX IF NOT EXISTS ix_lesson_plan_exports_lesson_plan_id "
        "ON lesson_plan_exports (lesson_plan_id)"
    )

    # -------------------------------------------------------------------------
    # 9. Recreate foreign keys with original ON DELETE behavior.
    # -------------------------------------------------------------------------
    fks = [
        ("users", "fk_users_school_id_schools", "school_id", "schools", "id", "SET NULL"),
        ("users", "fk_users_created_by_users", "created_by", "users", "id", "SET NULL"),
        ("sessions", "fk_sessions_user_id_users", "user_id", "users", "id", "CASCADE"),
        (
            "user_settings",
            "fk_user_settings_user_id_users",
            "user_id",
            "users",
            "id",
            "CASCADE",
        ),
        (
            "user_signup_profiles",
            "fk_user_signup_profiles_user_id_users",
            "user_id",
            "users",
            "id",
            "CASCADE",
        ),
        (
            "school_subjects",
            "fk_school_subjects_school_id_schools",
            "school_id",
            "schools",
            "id",
            "CASCADE",
        ),
        (
            "school_classes",
            "fk_school_classes_school_id_schools",
            "school_id",
            "schools",
            "id",
            "CASCADE",
        ),
        (
            "school_class_subjects",
            "fk_school_class_subjects_school_class_id",
            "school_class_id",
            "school_classes",
            "id",
            "CASCADE",
        ),
        (
            "school_class_subjects",
            "fk_school_class_subjects_subject_id",
            "subject_id",
            "school_subjects",
            "id",
            "CASCADE",
        ),
        (
            "textbook_uploads",
            "fk_textbook_uploads_uploaded_by_users",
            "uploaded_by",
            "users",
            "id",
            "SET NULL",
        ),
        (
            "textbook_images",
            "fk_textbook_images_textbook_upload_id",
            "textbook_upload_id",
            "textbook_uploads",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plans",
            "fk_lesson_plans_user_id_users",
            "user_id",
            "users",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plan_jobs",
            "fk_lesson_plan_jobs_lesson_plan_id",
            "lesson_plan_id",
            "lesson_plans",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plan_jobs",
            "fk_lesson_plan_jobs_user_id_users",
            "user_id",
            "users",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plan_artifacts",
            "fk_lesson_plan_artifacts_lesson_plan_id",
            "lesson_plan_id",
            "lesson_plans",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plan_artifacts",
            "fk_lesson_plan_artifacts_job_id",
            "job_id",
            "lesson_plan_jobs",
            "id",
            "SET NULL",
        ),
        (
            "lesson_plan_versions",
            "fk_lesson_plan_versions_lesson_plan_id",
            "lesson_plan_id",
            "lesson_plans",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plan_versions",
            "fk_lesson_plan_versions_created_by",
            "created_by",
            "users",
            "id",
            "SET NULL",
        ),
        (
            "lesson_plan_exports",
            "fk_lesson_plan_exports_lesson_plan_id",
            "lesson_plan_id",
            "lesson_plans",
            "id",
            "CASCADE",
        ),
        (
            "lesson_plan_exports",
            "fk_lesson_plan_exports_version_id",
            "version_id",
            "lesson_plan_versions",
            "id",
            "SET NULL",
        ),
    ]
    for table, name, col, ref_table, ref_col, ondelete in fks:
        _execute(
            f"""
            ALTER TABLE {_q(table)}
            ADD CONSTRAINT {_q(name)}
            FOREIGN KEY ({_q(col)})
            REFERENCES {_q(ref_table)} ({_q(ref_col)})
            ON DELETE {ondelete}
            """
        )


def downgrade() -> None:
    raise NotImplementedError(
        "UUID → BIGINT sequential PK migration is irreversible without a backup"
    )

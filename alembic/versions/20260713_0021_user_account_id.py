"""sequential account_id on users

Revision ID: 20260713_0021
Revises: 20260713_0020
Create Date: 2026-07-13 15:10:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "20260713_0021"
down_revision = "20260713_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("account_id", sa.BigInteger(), nullable=True))
    op.execute(
        """
        WITH numbered AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY created_at, id) AS rn
            FROM users
        )
        UPDATE users AS u
        SET account_id = n.rn
        FROM numbered AS n
        WHERE u.id = n.id
        """
    )
    op.execute("CREATE SEQUENCE IF NOT EXISTS users_account_id_seq OWNED BY users.account_id")
    op.execute(
        "SELECT setval('users_account_id_seq', COALESCE((SELECT MAX(account_id) FROM users), 0))"
    )
    op.alter_column(
        "users",
        "account_id",
        nullable=False,
        server_default=sa.text("nextval('users_account_id_seq')"),
    )
    op.create_index("ix_users_account_id", "users", ["account_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_account_id", table_name="users")
    op.drop_column("users", "account_id")
    op.execute("DROP SEQUENCE IF EXISTS users_account_id_seq")

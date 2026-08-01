from datetime import UTC, datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Identity, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.modules.auth.constants import Role


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(Enum(Role, name="user_role"), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    school_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("schools.id", ondelete="SET NULL"),
        nullable=True,
    )
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    teaching_board: Mapped[str | None] = mapped_column(String(100), nullable=True)
    teaching_subjects: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    # JSON: [{ "grade": "5", "sections": ["A", "B"] }, ...]; legacy: ["5","6"] still readable in API
    teaching_classes: Mapped[list[dict] | None] = mapped_column("teaching_grades", JSON, nullable=True)
    # Tutor: student ids excluded from auto class matching
    excluded_student_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    # Parked subjects/classes when teacher is deactivated (restored on activate)
    assignment_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Login credentials lifecycle (school-admin Credentials tab)
    credentials_generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    credentials_shared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    credential_delivery_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_by: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    username: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    language: Mapped[str] = mapped_column(String(32), nullable=False, default="en")
    theme: Mapped[str] = mapped_column(String(16), nullable=False, default="light")
    notify_email: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notify_push: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notify_assignments: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notify_sessions: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notify_messages: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

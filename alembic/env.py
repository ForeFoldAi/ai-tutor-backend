"""Alembic environment. Load `.env` before any `app` imports — `app.core.database` reads settings at import time."""
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent of this `alembic/` directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.core.database import Base
from app.modules.schools.models import School  # noqa: F401
from app.modules.sessions.models import SessionToken  # noqa: F401
from app.modules.users.models import User  # noqa: F401
from app.modules.catalog.models import BoardDefinition, SyllabusSubject, TextbookUpload  # noqa: F401
from app.modules.teacher.lesson_planner.models import (  # noqa: F401
    LessonArtifact,
    LessonExport,
    LessonJob,
    LessonPlan,
    LessonVersion,
)
from app.modules.school_admin.classes.models import SchoolClass, SchoolClassSubject, SchoolSubject  # noqa: F401
from app.modules.student_learning.models import (  # noqa: F401
    StudentChapterProgress,
    StudentLearningStreak,
    StudentStudySession,
)
from app.modules.live_sessions.models import LiveSessionJoin, LiveTutorSession  # noqa: F401
from app.modules.teacher.assignments.models import (  # noqa: F401
    AssignmentSubmission,
    TeacherAssignment,
)
from app.modules.auth.otp_models import PasswordResetOtp  # noqa: F401
from app.modules.notifications.models import Notification  # noqa: F401

config = context.config
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.modules.learning_intelligence.models import (  # noqa: F401
    LiaConcept,
    LiaConceptDependency,
    LiaConceptMastery,
    LiaKnowledgeGap,
    LiaLearningEvent,
    LiaLearningPrediction,
    LiaMetricHistory,
    LiaMisconception,
    LiaPeriodSummary,
    LiaRiskAssessment,
    LiaStudentProfile,
    LiaTeacherRecommendation,
    LiaTeachingMemory,
    LiaTutorGuidanceCache,
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

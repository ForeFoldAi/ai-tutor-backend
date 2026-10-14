from enum import StrEnum


class ArtifactType(StrEnum):
    LESSON_PLAN = "lesson_plan"
    TEACHING_NOTES = "teaching_notes"
    EXAMPLES = "examples"
    WORKSHEET = "worksheet"
    QUIZ = "quiz"
    HOMEWORK = "homework"
    PPT_OUTLINE = "ppt_outline"


class LessonPlanStatus(StrEnum):
    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactStatus(StrEnum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"


class ExportStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Celery queue names
QUEUE_GENERATE = "lesson-generate"
QUEUE_EXPORT = "lesson-export"
QUEUE_AUTOSAVE = "lesson-autosave"
QUEUE_REGENERATE = "lesson-regenerate"

# Redis key prefixes
REDIS_JOB_PREFIX = "lesson_planner:job:"
REDIS_CANCEL_PREFIX = "lesson_planner:cancel:"
REDIS_RATE_PREFIX = "lesson_planner:rate:"
REDIS_PUBSUB_CHANNEL = "lesson_planner:events:"

# WebSocket event types
WS_JOB_STARTED = "job_started"
WS_PROGRESS = "progress"
WS_ARTIFACT_STARTED = "artifact_started"
WS_ARTIFACT_COMPLETED = "artifact_completed"
WS_ARTIFACT_UPDATED = "artifact_updated"
WS_AUTOSAVE = "autosave"
WS_CANCELLED = "cancelled"
WS_ERROR = "error"
WS_COMPLETED = "completed"
WS_HEARTBEAT = "heartbeat"

ALL_ARTIFACT_TYPES = tuple(ArtifactType)

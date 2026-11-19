from enum import StrEnum


class AssignableArtifactType(StrEnum):
    WORKSHEET = "worksheet"
    QUIZ = "quiz"
    HOMEWORK = "homework"


class AssignmentStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"


class SubmissionStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    GRADED = "graded"

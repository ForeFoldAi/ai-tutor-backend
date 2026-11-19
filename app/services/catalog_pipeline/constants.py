"""Constants for the catalog and textbook ingestion Celery pipeline."""

QUEUE_CATALOG = "catalog-ingest"

TASK_PROCESS_UPLOAD = "catalog.process_upload"
TASK_ANALYZE_TEXTBOOK = "catalog.analyze_textbook"
TASK_PROCESS_CHAPTERS = "catalog.process_chapters"

# Processing safeguards
EMBEDDING_BATCH_SIZE = 50
MAX_INGESTION_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5
MAX_RETRY_BACKOFF_SECONDS = 60

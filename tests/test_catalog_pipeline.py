"""Unit tests for catalog Celery pipeline and dispatcher."""

import unittest
from unittest.mock import MagicMock, patch

from app.services.catalog_pipeline.constants import (
    EMBEDDING_BATCH_SIZE,
    QUEUE_CATALOG,
    TASK_ANALYZE_TEXTBOOK,
    TASK_PROCESS_CHAPTERS,
    TASK_PROCESS_UPLOAD,
)
from app.services.catalog_pipeline.dispatcher import (
    dispatch_textbook_analysis,
    dispatch_textbook_chapters,
    dispatch_upload_processing,
)


class TestCatalogPipeline(unittest.TestCase):
    def test_constants_and_batch_limits(self):
        self.assertEqual(QUEUE_CATALOG, "catalog-ingest")
        self.assertEqual(TASK_PROCESS_UPLOAD, "catalog.process_upload")
        self.assertEqual(TASK_ANALYZE_TEXTBOOK, "catalog.analyze_textbook")
        self.assertEqual(TASK_PROCESS_CHAPTERS, "catalog.process_chapters")
        self.assertGreaterEqual(EMBEDDING_BATCH_SIZE, 10)

    def test_celery_app_registration(self):
        from app.core.celery_app import celery_app

        # Queue registered in task_queues
        self.assertIn(QUEUE_CATALOG, celery_app.conf.task_queues)
        # Task routes registered
        self.assertEqual(celery_app.conf.task_routes.get(TASK_PROCESS_UPLOAD), {"queue": QUEUE_CATALOG})
        self.assertEqual(celery_app.conf.task_routes.get(TASK_ANALYZE_TEXTBOOK), {"queue": QUEUE_CATALOG})
        self.assertEqual(celery_app.conf.task_routes.get(TASK_PROCESS_CHAPTERS), {"queue": QUEUE_CATALOG})
        # Memory recycling safeguard present
        self.assertGreaterEqual(celery_app.conf.worker_max_tasks_per_child, 1)

    @patch("app.services.catalog_pipeline.dispatcher._can_use_celery", return_value=True)
    @patch("app.services.catalog_pipeline.tasks.process_upload_task.delay")
    def test_dispatch_upload_celery(self, mock_delay, _mock_can_use):
        mock_task = MagicMock()
        mock_task.id = "mock-task-123"
        mock_delay.return_value = mock_task

        dispatch_upload_processing(upload_id=42)
        mock_delay.assert_called_once_with(42)

    @patch("app.services.catalog_pipeline.dispatcher._can_use_celery", return_value=False)
    @patch("app.services.catalog_pipeline.handlers.upload_handler.handle_upload_processing")
    def test_dispatch_upload_fallback_background_tasks(self, _mock_handler, _mock_can_use):
        bg_tasks = MagicMock()
        dispatch_upload_processing(upload_id=42, background_tasks=bg_tasks)
        bg_tasks.add_task.assert_called_once()

    @patch("app.services.catalog_pipeline.dispatcher._can_use_celery", return_value=True)
    @patch("app.services.catalog_pipeline.tasks.analyze_textbook_task.delay")
    def test_dispatch_textbook_analysis_celery(self, mock_delay, _mock_can_use):
        mock_task = MagicMock()
        mock_task.id = "mock-analysis-456"
        mock_delay.return_value = mock_task

        dispatch_textbook_analysis(textbook_id=99)
        mock_delay.assert_called_once_with(99)

    @patch("app.services.catalog_pipeline.dispatcher._can_use_celery", return_value=True)
    @patch("app.services.catalog_pipeline.tasks.process_chapters_task.delay")
    def test_dispatch_textbook_chapters_celery(self, mock_delay, _mock_can_use):
        mock_task = MagicMock()
        mock_task.id = "mock-chapters-789"
        mock_delay.return_value = mock_task

        dispatch_textbook_chapters(textbook_id=99)
        mock_delay.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()

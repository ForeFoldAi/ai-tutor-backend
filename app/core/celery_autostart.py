"""Optional local sidecar: start Celery worker + beat with the API process."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PROCS: list[subprocess.Popen] = []
_ROOT = Path(__file__).resolve().parent.parent.parent


def start_celery_sidecar() -> None:
    """Spawn worker + beat if CELERY_AUTOSTART is on. No-op if already running."""
    global _PROCS
    settings = get_settings()
    if not settings.celery_autostart:
        return
    if _PROCS:
        return

    py = sys.executable
    env = os.environ.copy()
    # Avoid nested reload weirdness in children
    env.pop("WERKZEUG_RUN_MAIN", None)

    worker_cmd = [
        py,
        "-m",
        "celery",
        "-A",
        "app.core.celery_app.celery_app",
        "worker",
        "-Q",
        "lesson-generate,lesson-export,lesson-autosave,lesson-regenerate,mail",
        "-l",
        "info",
    ]
    beat_cmd = [
        py,
        "-m",
        "celery",
        "-A",
        "app.core.celery_app.celery_app",
        "beat",
        "-l",
        "info",
    ]

    try:
        worker = subprocess.Popen(
            worker_cmd,
            cwd=str(_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        beat = subprocess.Popen(
            beat_cmd,
            cwd=str(_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        logger.exception("CELERY_AUTOSTART failed to spawn worker/beat")
        stop_celery_sidecar()
        return

    _PROCS = [worker, beat]
    logger.info(
        "CELERY_AUTOSTART: worker pid=%s beat pid=%s (sessions %02d:%02d, assignments %02d:%02d %s)",
        worker.pid,
        beat.pid,
        settings.mail_session_reminder_hour,
        settings.mail_reminder_minute,
        settings.mail_assignment_reminder_hour,
        settings.mail_reminder_minute,
        settings.mail_reminder_tz,
    )


def stop_celery_sidecar() -> None:
    global _PROCS
    for proc in _PROCS:
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.terminate()
            except Exception:
                pass
    for proc in _PROCS:
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    if _PROCS:
        logger.info("CELERY_AUTOSTART: stopped worker/beat")
    _PROCS = []

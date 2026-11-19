"""Alert fan-out: email, Telegram, Slack, in-admin notifications."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def dispatch_alert_channels(
    db: Session,
    *,
    channels: list[str] | None,
    title: str,
    message: str,
    severity: str,
    settings_row: Any | None = None,
    recipient_user_ids: list[int] | None = None,
) -> dict[str, bool]:
    channels = channels or ["in_app", "email"]
    results: dict[str, bool] = {}
    if "email" in channels:
        results["email"] = _send_email(title, message, settings_row)
    if "telegram" in channels:
        results["telegram"] = _send_telegram(title, message, settings_row)
    if "slack" in channels:
        results["slack"] = _send_slack(title, message, severity, settings_row)
    if "in_app" in channels:
        results["in_app"] = _send_in_app(db, title, message, recipient_user_ids or [])
    return results


def _send_email(title: str, message: str, settings_row: Any | None) -> bool:
    try:
        from app.core.config import get_settings
        from app.services.mail.smtp import send_message

        settings = get_settings()
        to_addr = (settings_row.alert_email_to if settings_row else None) or settings.admin_notify_email
        if not to_addr or not settings.smtp_enabled:
            return False
        return bool(
            send_message(
                to=to_addr,
                subject=f"[Alert] {title}",
                text_body=message,
            )
        )
    except Exception as exc:
        logger.warning("alert email failed: %s", exc)
        return False


def severity_tag(severity: str) -> str:
    return severity.upper()


def _send_telegram(title: str, message: str, settings_row: Any | None) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = (settings_row.telegram_chat_id if settings_row else None) or os.environ.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat_id:
        return False
    text = f"*{title}*\n{message}"[:3500]
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            )
            return resp.status_code < 300
    except Exception as exc:
        logger.warning("telegram alert failed: %s", exc)
        return False


def _send_slack(title: str, message: str, severity: str, settings_row: Any | None) -> bool:
    url = (settings_row.slack_webhook_url if settings_row else None) or os.environ.get("SLACK_WEBHOOK_URL") or ""
    if not url:
        return False
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(
                url,
                json={
                    "text": f"[{severity}] {title}",
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": message[:2800]}},
                    ],
                },
            )
            return resp.status_code < 300
    except Exception as exc:
        logger.warning("slack alert failed: %s", exc)
        return False


def _send_in_app(db: Session, title: str, message: str, user_ids: list[int]) -> bool:
    if not user_ids:
        try:
            from app.modules.auth.constants import Role
            from app.modules.users.models import User

            user_ids = [
                int(r[0])
                for r in db.query(User.id).filter(User.role == Role.MASTER_ADMIN, User.is_active.is_(True)).all()
            ]
        except Exception:
            return False
    try:
        from app.modules.notifications.models import Notification

        for uid in user_ids:
            db.add(
                Notification(
                    recipient_user_id=uid,
                    type="monitoring_alert",
                    title=title[:200],
                    body=message[:2000],
                    link="/master-admin/monitor/alerts",
                )
            )
        db.commit()
        return True
    except Exception as exc:
        logger.warning("in-app alert failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return False

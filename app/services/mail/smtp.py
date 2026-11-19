"""Async SMTP send via aiosmtplib (sync wrapper for FastAPI sync routes)."""

from __future__ import annotations

import asyncio
import logging
from email.message import EmailMessage
from email.utils import formataddr

import aiosmtplib

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def send_message_async(
    *,
    to: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
) -> bool:
    """Send one email. Returns True on success / intentional no-op; False on failure."""
    settings = get_settings()
    if not settings.smtp_enabled:
        logger.info(
            "SMTP disabled; skip send to=%s subject=%s",
            to,
            subject,
        )
        return True

    msg = EmailMessage()
    msg["From"] = formataddr((settings.app_name, settings.smtp_from))
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user or None,
            password=settings.smtp_password or None,
            start_tls=settings.smtp_tls,
            validate_certs=settings.smtp_validate_certs,
        )
        return True
    except Exception:
        logger.exception("SMTP send failed to=%s subject=%s", to, subject)
        return False


def send_message(
    *,
    to: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
) -> bool:
    """Sync wrapper. Safe from sync FastAPI handlers (not nested event loops)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # ponytail: nested loop ceiling — sync routes are the normal path; upgrade to
        # await send_message_async from async routes if callers move to async def.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                send_message_async(
                    to=to,
                    subject=subject,
                    text_body=text_body,
                    html_body=html_body,
                ),
            ).result()
    return asyncio.run(
        send_message_async(
            to=to,
            subject=subject,
            text_body=text_body,
            html_body=html_body,
        )
    )

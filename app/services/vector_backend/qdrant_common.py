"""Shared Qdrant client + filter helpers."""

from __future__ import annotations

import hashlib
import logging
import uuid
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

TEXT_VECTOR_SIZE = 768  # BAAI/bge-base-en-v1.5
IMAGE_VECTOR_SIZE = 512  # openai/clip-vit-base-patch32


def _qdrant_timeout() -> float:
    try:
        from app.config import QDRANT_TIMEOUT

        return float(QDRANT_TIMEOUT)
    except Exception:
        import os

        return float(os.environ.get("QDRANT_TIMEOUT", "60") or "60")


@lru_cache(maxsize=1)
def get_qdrant_client():
    from urllib.parse import urlparse

    from qdrant_client import QdrantClient

    from app.config import QDRANT_API_KEY, QDRANT_URL

    timeout = _qdrant_timeout()
    raw = (QDRANT_URL or "http://localhost:6333").rstrip("/")
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    kwargs: dict[str, Any] = {
        "url": raw,
        "timeout": timeout,
        "prefer_grpc": False,
        "check_compatibility": False,
    }
    # qdrant-client defaults to :6333 even for https://host with no port,
    # which breaks public HTTPS frontends on :443 (ConnectTimeout).
    if parsed.scheme == "https" and parsed.port is None:
        kwargs["https"] = True
        kwargs["port"] = 443
    elif parsed.scheme == "http" and parsed.port is None:
        kwargs["port"] = 6333
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    logger.info(
        "[QDRANT] client url=%s port=%s https=%s timeout=%ss",
        kwargs.get("url"),
        kwargs.get("port"),
        kwargs.get("https"),
        timeout,
    )
    return QdrantClient(**kwargs)


def _is_not_found(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "not found" in msg
        or "404" in msg
        or "doesn't exist" in msg
        or "does not exist" in msg
        or "notfound" in name
    )


def _is_already_exists(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "already exists" in msg or "conflict" in msg or "409" in msg


def _is_connectivity(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "timed out",
        "timeout",
        "connect",
        "unreachable",
        "name or service not known",
        "connection refused",
        "network",
        "403 forbidden",
    )
    return any(n in msg for n in needles) or any(n in name for n in ("timeout", "connect"))


def ensure_collection(name: str, *, vector_size: int, distance: str = "Cosine") -> None:
    """
    Ensure a Qdrant collection exists; create it if missing.

    Distinguishes "not found" (auto-create) from "unreachable" (clear error).
    """
    from qdrant_client.http import models as qm

    from app.config import QDRANT_URL

    client = get_qdrant_client()

    # 1) Already present?
    try:
        if hasattr(client, "collection_exists") and callable(client.collection_exists):
            if client.collection_exists(name):
                return
        else:
            client.get_collection(name)
            return
    except Exception as exc:
        if _is_connectivity(exc):
            raise RuntimeError(
                f"Qdrant unreachable at {QDRANT_URL!r} while ensuring collection "
                f"{name!r}. Auto-create needs a reachable Qdrant (check VPN/network/"
                f"QDRANT_URL). Underlying: {exc}"
            ) from exc
        if not _is_not_found(exc):
            logger.warning(
                "[QDRANT] existence check for %s failed (%s); attempting create",
                name,
                exc,
            )

    # 2) Auto-create
    dist = getattr(qm.Distance, distance, qm.Distance.COSINE)
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=dist),
        )
        logger.info(
            "[QDRANT] auto-created collection %s size=%d %s",
            name,
            vector_size,
            distance,
        )
    except Exception as exc:
        if _is_already_exists(exc):
            logger.info("[QDRANT] collection %s already exists (race)", name)
            return
        if _is_connectivity(exc):
            raise RuntimeError(
                f"Qdrant unreachable at {QDRANT_URL!r}; cannot auto-create collection "
                f"{name!r}. Fix connectivity to {QDRANT_URL}. Underlying: {exc}"
            ) from exc
        raise


def textbook_upload_filter(upload_ids: list[str]):
    """Qdrant Filter matching payload textbook_upload_id (sequential BIGINT strings)."""
    from qdrant_client.http import models as qm

    ids = [str(x).strip() for x in upload_ids if str(x).strip()]
    if not ids:
        return None
    if len(ids) == 1:
        return qm.Filter(
            must=[qm.FieldCondition(key="textbook_upload_id", match=qm.MatchValue(value=ids[0]))]
        )
    return qm.Filter(
        must=[qm.FieldCondition(key="textbook_upload_id", match=qm.MatchAny(any=ids))]
    )


def deterministic_text_point_id(upload_id: str, page: Any, ordinal: int, text: str) -> str:
    """Stable UUID5 from sequential upload id + page + ordinal (not a public entity UUID)."""
    digest = hashlib.sha256((text or "")[:200].encode("utf-8", errors="ignore")).hexdigest()[:12]
    key = f"text:{upload_id}:{page}:{ordinal}:{digest}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def image_point_id(image_id: str) -> int | str:
    """Prefer sequential BIGINT as unsigned point id; fall back to UUID5."""
    try:
        n = int(str(image_id).strip())
        if n >= 0:
            return n
    except (TypeError, ValueError):
        pass
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"image:{image_id}"))

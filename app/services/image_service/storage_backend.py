"""
Storage backend abstraction for textbook figure images.

Supports:
  - LocalStorageBackend  — current file-system store (default)
  - S3StorageBackend     — AWS S3 / S3-compatible (MinIO, GCS via interop)

Configuration (app/config.py → .env):
  STORAGE_BACKEND = "local" | "s3"
  S3_BUCKET        — bucket name (required for s3)
  S3_ENDPOINT_URL  — override endpoint (optional; for MinIO / GCS)
  S3_REGION        — region (default "ap-south-1")
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY — credentials (or use IAM role)
  CDN_BASE_URL     — CDN prefix for public URLs (optional)
  IMAGE_THUMB_SIZE — (width, height) for thumbnail generation, e.g. "320x240"

Design principles:
  - All paths stored in DB are RELATIVE (upload_id/filename) — backend-agnostic.
  - `get_url()` translates relative paths to absolute URLs at request time.
  - Deduplication via SHA-256 hash before saving (skip duplicate bytes).
  - Thumbnails generated at save time if PIL available and IMAGE_THUMB_SIZE set.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StorageBackend(ABC):
    """Interface for image persistence backends."""

    @abstractmethod
    def save(self, relative_path: str, data: bytes) -> tuple[str, str]:
        """
        Save *data* at *relative_path*.

        Returns (stored_path, content_hash) where stored_path == relative_path
        on success, or the path of an existing duplicate when deduplicated.
        """

    @abstractmethod
    def delete(self, relative_path: str) -> None:
        """Remove an object; no-op if it doesn't exist."""

    @abstractmethod
    def exists(self, relative_path: str) -> bool:
        """Return True if the object exists in the store."""

    @abstractmethod
    def get_url(self, relative_path: str) -> str:
        """Return a public-facing URL for the stored object."""

    @abstractmethod
    def get_bytes(self, relative_path: str) -> Optional[bytes]:
        """Return raw bytes for the object, or None if not found."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _make_thumbnail(data: bytes, size: tuple[int, int]) -> bytes | None:
        try:
            from PIL import Image

            with Image.open(BytesIO(data)) as im:
                im.thumbnail(size, Image.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=75, optimize=True)
                return buf.getvalue()
        except Exception as exc:
            logger.debug("Thumbnail generation failed: %s", exc)
            return None

    def _thumb_relative_path(self, relative_path: str) -> str:
        base, ext = os.path.splitext(relative_path)
        return f"{base}_thumb{ext}"


# ---------------------------------------------------------------------------
# Local file-system backend (current behaviour, no behaviour change)
# ---------------------------------------------------------------------------

class LocalStorageBackend(StorageBackend):
    """
    Store images on the local file system under *root_dir*.

    *root_dir* should be set to UPLOADS_DIR/_textbook_images (see config).
    """

    def __init__(self, root_dir: str, *, cdn_base_url: str = "", thumb_size: tuple[int, int] | None = None) -> None:
        self.root_dir = root_dir
        self.cdn_base_url = (cdn_base_url or "").rstrip("/")
        self.thumb_size = thumb_size

    def _abs(self, relative_path: str) -> str:
        return os.path.join(self.root_dir, relative_path)

    def save(self, relative_path: str, data: bytes) -> tuple[str, str]:
        content_hash = self.compute_hash(data)
        abs_path = self._abs(relative_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # Skip write if file already exists with same hash (deduplication)
        if os.path.isfile(abs_path):
            existing_hash = self.compute_hash(open(abs_path, "rb").read())
            if existing_hash == content_hash:
                logger.debug("[STORAGE] dedup hit: %s", relative_path)
                return relative_path, content_hash

        with open(abs_path, "wb") as f:
            f.write(data)

        if self.thumb_size:
            thumb = self._make_thumbnail(data, self.thumb_size)
            if thumb:
                thumb_path = self._abs(self._thumb_relative_path(relative_path))
                with open(thumb_path, "wb") as f:
                    f.write(thumb)

        return relative_path, content_hash

    def delete(self, relative_path: str) -> None:
        for path in [self._abs(relative_path), self._abs(self._thumb_relative_path(relative_path))]:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                logger.debug("[STORAGE] delete error %s: %s", path, exc)

    def exists(self, relative_path: str) -> bool:
        return os.path.isfile(self._abs(relative_path))

    def get_url(self, relative_path: str) -> str:
        if self.cdn_base_url:
            return f"{self.cdn_base_url}/{relative_path}"
        # Fall back to API path (served by FastAPI route)
        upload_id, fname = relative_path.split("/", 1) if "/" in relative_path else ("", relative_path)
        return f"/auth/catalog/textbook-images/{upload_id}/{fname}"

    def get_bytes(self, relative_path: str) -> bytes | None:
        path = self._abs(relative_path)
        if not os.path.isfile(path):
            return None
        try:
            return open(path, "rb").read()
        except OSError:
            return None


# ---------------------------------------------------------------------------
# S3 / S3-compatible backend
# ---------------------------------------------------------------------------

class S3StorageBackend(StorageBackend):
    """
    Store images in an S3 bucket (or S3-compatible store such as MinIO / GCS).

    Requires boto3 (`pip install boto3`).
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "textbook_images",
        region: str = "ap-south-1",
        endpoint_url: str | None = None,
        cdn_base_url: str = "",
        thumb_size: tuple[int, int] | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cdn_base_url = (cdn_base_url or "").rstrip("/")
        self.thumb_size = thumb_size
        try:
            import boto3

            session_kwargs: dict = {}
            if access_key and secret_key:
                session_kwargs["aws_access_key_id"] = access_key
                session_kwargs["aws_secret_access_key"] = secret_key
            session = boto3.Session(**session_kwargs)
            client_kwargs: dict = {"region_name": region}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            self._client = session.client("s3", **client_kwargs)
            logger.info("[STORAGE] S3 backend ready (bucket=%s prefix=%s)", bucket, prefix)
        except ImportError as exc:
            raise ImportError("S3StorageBackend requires boto3: pip install boto3") from exc

    def _key(self, relative_path: str) -> str:
        return f"{self.prefix}/{relative_path}" if self.prefix else relative_path

    def save(self, relative_path: str, data: bytes) -> tuple[str, str]:
        content_hash = self.compute_hash(data)
        key = self._key(relative_path)

        # Check for existing object with same hash (via ETag or metadata)
        try:
            head = self._client.head_object(Bucket=self.bucket, Key=key)
            stored_hash = head.get("Metadata", {}).get("content-hash", "")
            if stored_hash == content_hash:
                logger.debug("[STORAGE] S3 dedup hit: %s", key)
                return relative_path, content_hash
        except Exception:
            pass  # Object not found or other error — proceed with upload

        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType="image/jpeg",
            Metadata={"content-hash": content_hash},
        )

        if self.thumb_size:
            thumb = self._make_thumbnail(data, self.thumb_size)
            if thumb:
                thumb_key = self._key(self._thumb_relative_path(relative_path))
                self._client.put_object(
                    Bucket=self.bucket,
                    Key=thumb_key,
                    Body=thumb,
                    ContentType="image/jpeg",
                )

        return relative_path, content_hash

    def delete(self, relative_path: str) -> None:
        for key in [self._key(relative_path), self._key(self._thumb_relative_path(relative_path))]:
            try:
                self._client.delete_object(Bucket=self.bucket, Key=key)
            except Exception as exc:
                logger.debug("[STORAGE] S3 delete error %s: %s", key, exc)

    def exists(self, relative_path: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(relative_path))
            return True
        except Exception:
            return False

    def get_url(self, relative_path: str) -> str:
        if self.cdn_base_url:
            return f"{self.cdn_base_url}/{self._key(relative_path)}"
        # Pre-signed URL (expires in 1 hour)
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self._key(relative_path)},
                ExpiresIn=3600,
            )
        except Exception as exc:
            logger.warning("[STORAGE] presign failed for %s: %s", relative_path, exc)
            return f"s3://{self.bucket}/{self._key(relative_path)}"

    def get_bytes(self, relative_path: str) -> bytes | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._key(relative_path))
            return resp["Body"].read()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_backend_instance: StorageBackend | None = None


def _parse_thumb_size(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        parts = raw.lower().replace("x", " ").split()
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def get_storage_backend() -> StorageBackend:
    """
    Return the configured storage backend singleton.

    Configuration is read from app.config on first call.
    """
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    from app.config import UPLOADS_DIR

    try:
        from app.config import (
            STORAGE_BACKEND,
            S3_BUCKET,
            S3_ENDPOINT_URL,
            S3_REGION,
            CDN_BASE_URL,
            IMAGE_THUMB_SIZE,
            AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY,
        )
    except ImportError:
        # Graceful fallback: config vars not yet added
        STORAGE_BACKEND = "local"
        S3_BUCKET = ""
        S3_ENDPOINT_URL = ""
        S3_REGION = "ap-south-1"
        CDN_BASE_URL = ""
        IMAGE_THUMB_SIZE = ""
        AWS_ACCESS_KEY_ID = ""
        AWS_SECRET_ACCESS_KEY = ""

    thumb_size = _parse_thumb_size(IMAGE_THUMB_SIZE)

    backend_name = (STORAGE_BACKEND or "local").lower()

    if backend_name == "s3":
        if not S3_BUCKET:
            logger.warning("[STORAGE] STORAGE_BACKEND=s3 but S3_BUCKET not set; falling back to local")
        else:
            try:
                _backend_instance = S3StorageBackend(
                    bucket=S3_BUCKET,
                    region=S3_REGION or "ap-south-1",
                    endpoint_url=S3_ENDPOINT_URL or None,
                    cdn_base_url=CDN_BASE_URL or "",
                    thumb_size=thumb_size,
                    access_key=AWS_ACCESS_KEY_ID or None,
                    secret_key=AWS_SECRET_ACCESS_KEY or None,
                )
                return _backend_instance
            except Exception as exc:
                logger.warning("[STORAGE] S3 backend init failed, using local: %s", exc)

    # Default: local
    image_root = os.path.join(UPLOADS_DIR, "_textbook_images")
    _backend_instance = LocalStorageBackend(
        root_dir=image_root,
        cdn_base_url=CDN_BASE_URL if "CDN_BASE_URL" in dir() else "",
        thumb_size=thumb_size,
    )
    logger.info("[STORAGE] Using local backend at %s", image_root)
    return _backend_instance
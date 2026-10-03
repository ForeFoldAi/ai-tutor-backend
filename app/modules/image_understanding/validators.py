"""Production validation for student-uploaded educational images."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import (
    IMAGE_ALLOWED_MIME,
    IMAGE_MAX_HEIGHT,
    IMAGE_MAX_SIZE_MB,
    IMAGE_MAX_WIDTH,
)
from app.modules.image_understanding.errors import ImageTooLargeError, InvalidImageError

logger = logging.getLogger(__name__)

_EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    mime_type: str
    extension: str
    width: int
    height: int
    content_hash: str


def _allowed_mimes() -> set[str]:
    out = set()
    for m in IMAGE_ALLOWED_MIME:
        m = m.lower().strip()
        if m == "image/jpg":
            out.add("image/jpeg")
        out.add(m)
    return out or {"image/jpeg", "image/png", "image/webp"}


def _sniff_mime(data: bytes) -> str | None:
    if len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _ext_from_filename(filename: str | None) -> str | None:
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower().strip()


def validate_image_bytes(
    data: bytes,
    *,
    filename: str | None = None,
    declared_mime: str | None = None,
) -> ValidatedImage:
    import hashlib

    from PIL import Image

    if not data:
        raise InvalidImageError("Please upload a valid JPG, PNG, or WEBP image.")

    max_bytes = int(IMAGE_MAX_SIZE_MB * 1024 * 1024)
    if len(data) > max_bytes:
        raise ImageTooLargeError(
            f"That image is too large. Please upload an image under {IMAGE_MAX_SIZE_MB:g} MB."
        )

    allowed = _allowed_mimes()
    sniffed = _sniff_mime(data)
    if sniffed is None:
        raise InvalidImageError()

    if sniffed not in allowed and sniffed.replace("image/jpg", "image/jpeg") not in allowed:
        raise InvalidImageError()

    ext = _ext_from_filename(filename)
    if ext and ext not in _EXT_TO_MIME:
        raise InvalidImageError()
    if ext and _EXT_TO_MIME[ext] not in (sniffed, "image/jpeg" if sniffed == "image/jpeg" else sniffed):
        # Extension must match sniffed type when provided
        if _EXT_TO_MIME.get(ext) != sniffed:
            raise InvalidImageError()

    declared = (declared_mime or "").split(";")[0].strip().lower()
    if declared in ("image/jpg",):
        declared = "image/jpeg"
    if declared and declared not in allowed:
        # Do not trust client MIME alone, but reject clearly wrong declared types
        if declared.startswith("image/") and declared != sniffed:
            raise InvalidImageError()
        if not declared.startswith("image/"):
            raise InvalidImageError()

    # Decompression bomb guard
    Image.MAX_IMAGE_PIXELS = max(IMAGE_MAX_WIDTH * IMAGE_MAX_HEIGHT * 2, 50_000_000)

    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as im:
            im.verify()
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            if width < 1 or height < 1:
                raise InvalidImageError()
            if width > IMAGE_MAX_WIDTH or height > IMAGE_MAX_HEIGHT:
                # Soft: preprocessing will resize; only hard-reject absurd sizes
                if width > IMAGE_MAX_WIDTH * 4 or height > IMAGE_MAX_HEIGHT * 4:
                    raise ImageTooLargeError(
                        "That image's dimensions are too large. Please upload a smaller photo."
                    )
    except ImageTooLargeError:
        raise
    except InvalidImageError:
        raise
    except Exception:
        logger.info("Image decode validation failed (details omitted)")
        raise InvalidImageError("That file looks corrupted. Please upload a clearer photo.") from None

    mime = sniffed
    extension = _MIME_TO_EXT.get(mime, "jpg")
    content_hash = hashlib.sha256(data).hexdigest()
    return ValidatedImage(
        data=data,
        mime_type=mime,
        extension=extension,
        width=width,
        height=height,
        content_hash=content_hash,
    )

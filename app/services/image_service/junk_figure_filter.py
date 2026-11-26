"""
Detect extracted "figures" that are not teaching figures: QR codes, chapter-opener
banners carrying a QR code, and decorations that repeat across chapters of one book
(mascots, page-margin art, "?" icons, recurring think-boxes).

Board/class/subject agnostic — rules use only the image itself, its figure number and
the other chapters of the same book, never textbook-specific names or coordinates.

Rules (each verified on CBSE Class 10, 960 figures, zero false positives):

* ``qr_code`` / ``qr_banner`` — a QR code that actually *decodes*. Finder-pattern
  detection alone (``QRCodeDetector.detect``) flags ~50% non-QR diagrams, so it is never
  used on its own. Figures with a printed figure number are kept: a real "Fig. X" never
  carries a QR code, so a hit there is more likely a misread than junk.
* ``recurring_decoration`` — near-identical image (pHash Hamming ≤ 6) present in at least
  3 different uploads of the same book, and no figure number on any copy.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REASON_QR_CODE = "qr_code"
REASON_QR_BANNER = "qr_banner"
REASON_RECURRING = "recurring_decoration"

# QR polygon covering at least this share of the crop → the crop *is* the QR code.
QR_STANDALONE_MIN_AREA = 0.35
# Small crops are upscaled before decoding (NCERT chapter QR codes are ~145 px).
_QR_UPSCALE_BELOW_PX = 600

RECURRING_MIN_UPLOADS = 3
RECURRING_MAX_HAMMING = 6


@dataclass(frozen=True)
class QrHit:
    payload: str
    area_fraction: float


def decode_qr_codes(image_bytes: bytes) -> list[QrHit]:
    """Decoded QR codes in an image. Empty when none decode or OpenCV is unavailable."""
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover - cv2 is a hard dependency in prod images
        return []
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return []
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return []

    detector = cv2.QRCodeDetector()
    scales = (1, 2, 3) if max(h, w) < _QR_UPSCALE_BELOW_PX else (1,)
    for scale in scales:
        probe = img if scale == 1 else cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        try:
            ok, payloads, points, _ = detector.detectAndDecodeMulti(probe)
        except Exception:
            continue
        if not ok or points is None:
            continue
        hits: list[QrHit] = []
        for payload, poly in zip(payloads, points):
            if not payload:
                continue
            area = float(cv2.contourArea((poly / scale).astype(np.float32)))
            hits.append(QrHit(payload=str(payload), area_fraction=area / float(w * h)))
        if hits:
            return hits
    return []


def classify_qr_junk(image_bytes: bytes, *, figure_number: str | None) -> str | None:
    """``qr_code`` / ``qr_banner`` when the crop carries a decodable QR code, else None."""
    if (figure_number or "").strip():
        return None
    hits = decode_qr_codes(image_bytes)
    if not hits:
        return None
    if max(h.area_fraction for h in hits) >= QR_STANDALONE_MIN_AREA:
        return REASON_QR_CODE
    return REASON_QR_BANNER


def phash_distance(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFFFFFFFFFF) ^ (b & 0xFFFFFFFFFFFFFFFF)).bit_count()


@dataclass(frozen=True)
class BookFigure:
    """One figure of a book, for cross-chapter recurrence detection."""

    key: Hashable
    upload_id: int
    phash: int | None
    figure_number: str | None


def find_recurring_decorations(
    figures: Iterable[BookFigure],
    *,
    min_uploads: int = RECURRING_MIN_UPLOADS,
    max_hamming: int = RECURRING_MAX_HAMMING,
) -> set[Hashable]:
    """Keys of unnumbered figures that repeat across ``min_uploads`` uploads of one book.

    Pass the figures of a single book (same board, class and subject). A group is only
    flagged when none of its copies carries a figure number.
    """
    items = [f for f in figures if f.phash is not None]
    # Connected components over "near-duplicate" edges, so a decoration whose copies drift
    # slightly page to page (crop offsets, page numbers inside the art) stays one group.
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if phash_distance(items[i].phash, items[j].phash) <= max_hamming:  # type: ignore[arg-type]
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    flagged: set[Hashable] = set()
    for members in groups.values():
        if len({items[j].upload_id for j in members}) < min_uploads:
            continue
        if any((items[j].figure_number or "").strip() for j in members):
            continue
        flagged.update(items[j].key for j in members)
    return flagged


def matching_sibling_keys(
    phash: int | None,
    siblings: Iterable[BookFigure],
    *,
    max_hamming: int = RECURRING_MAX_HAMMING,
) -> tuple[set[int], list[Hashable]]:
    """Uploads and keys of sibling-chapter figures that look like ``phash``."""
    if phash is None:
        return set(), []
    uploads: set[int] = set()
    keys: list[Hashable] = []
    for s in siblings:
        if s.phash is None or phash_distance(phash, s.phash) > max_hamming:
            continue
        uploads.add(s.upload_id)
        keys.append(s.key)
    return uploads, keys

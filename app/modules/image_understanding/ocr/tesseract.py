"""Tesseract wrapper: one image_to_data call → validated OCRResult (words, boxes, confidences)."""

from __future__ import annotations

import functools
import logging
import os
import re
import shutil
import time
from io import BytesIO
from typing import Any

from app.config import TESSERACT_CMD, TESSERACT_LANG, TESSERACT_TIMEOUT_SEC
from app.modules.image_understanding.ocr.schemas import OCRResult, OCRWord

logger = logging.getLogger(__name__)

# Tokens kept even without letters/digits: they carry math meaning for routing.
_MATH_SYMBOLS = set("=+-−×÷*/^√<>≤≥%π∫∑()")
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
# Ink coverage: edge pixels stronger than this count as ink.
_COVERAGE_MAX_SIDE = 1024
_INK_EDGE = 60
_BOX_PAD = 3
_COVERAGE_MIN_WORD_CONF = 0.5
_LOW_WORD_CONF = 0.6


def _keep_token(text: str) -> bool:
    if not text:
        return False
    if _ALNUM.search(text):
        return True
    return all(c in _MATH_SYMBOLS for c in text)


def parse_image_to_data(data: dict[str, list[Any]], *, language: str) -> OCRResult:
    """Turn pytesseract Output.DICT into an OCRResult (text_coverage filled by ink_text_coverage)."""
    n = len(data.get("text") or [])
    words: list[OCRWord] = []
    lines: dict[tuple[int, int, int], list[str]] = {}

    def col(key: str, i: int, default: Any = 0) -> Any:
        seq = data.get(key) or []
        return seq[i] if i < len(seq) else default

    for i in range(n):
        try:
            left, top = max(0, int(col("left", i))), max(0, int(col("top", i)))
            w, h = max(0, int(col("width", i))), max(0, int(col("height", i)))
            conf = float(col("conf", i, -1))
            key = (int(col("block_num", i)), int(col("par_num", i)), int(col("line_num", i)))
        except (TypeError, ValueError):
            continue
        word = OCRWord(text=col("text", i, ""), confidence=conf / 100.0, left=left, top=top, width=w, height=h)
        if conf < 0 or not _keep_token(word.text):
            continue
        words.append(word)
        lines.setdefault(key, []).append(word.text)

    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    avg = sum(w.confidence for w in words) / len(words) if words else 0.0
    low = sum(w.confidence < _LOW_WORD_CONF for w in words) / len(words) if words else 0.0
    meaningful = any(len(_ALNUM.findall(w.text)) >= 2 for w in words)
    return OCRResult(
        text=text,
        words=words,
        average_confidence=avg,
        low_confidence_ratio=low,
        word_count=len(words),
        has_text=meaningful,
        language=language,
    )


def ink_text_coverage(image, words: list[OCRWord]) -> float:
    """Share of visible ink (strokes, lines, drawings) that lies inside confidently-read word boxes.

    ~1.0 for a page of printed text; low when diagrams, graphs, maps, photos or handwriting carry
    ink Tesseract did not read. Page margins / background do not count against it.
    """
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    gray = image.convert("L")
    scale = min(1.0, _COVERAGE_MAX_SIDE / max(gray.size))
    if scale < 1.0:
        gray = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))))
    # Edges, not darkness: independent of paper/desk/board colour; flat regions contribute nothing.
    ink = gray.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > _INK_EDGE else 0)
    # FIND_EDGES fires along the image frame itself; blank a thin border.
    ImageDraw.Draw(ink).rectangle((0, 0, ink.width - 1, ink.height - 1), outline=0, width=2)
    total = ink.histogram()[255]
    if total == 0:
        return 0.0
    mask = Image.new("L", gray.size, 0)
    draw = ImageDraw.Draw(mask)
    for w in words:
        if w.confidence < _COVERAGE_MIN_WORD_CONF:
            continue
        draw.rectangle(
            (
                w.left * scale - _BOX_PAD,
                w.top * scale - _BOX_PAD,
                (w.left + w.width) * scale + _BOX_PAD,
                (w.top + w.height) * scale + _BOX_PAD,
            ),
            fill=255,
        )
    inside = ImageChops.multiply(ink, mask).histogram()[255]
    return inside / total


class TesseractOCRService:
    def __init__(
        self,
        *,
        lang: str = TESSERACT_LANG,
        timeout: float = TESSERACT_TIMEOUT_SEC,
        cmd: str = TESSERACT_CMD,
    ) -> None:
        self.lang = lang
        self.timeout = timeout
        self.cmd = cmd

    def _pytesseract(self):
        import pytesseract

        # Module-global in pytesseract: always set it so one service's cmd never leaks into another.
        pytesseract.pytesseract.tesseract_cmd = self.cmd or "tesseract"
        return pytesseract

    def run(self, image_bytes: bytes) -> OCRResult:
        t0 = time.perf_counter()

        def failed(code: str) -> OCRResult:
            return OCRResult(
                error=code,
                language=self.lang,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            pt = self._pytesseract()
        except ImportError:
            return failed("pytesseract_not_installed")
        try:
            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as im:
                rgb = im.convert("RGB")
            data = pt.image_to_data(rgb, lang=self.lang, output_type=pt.Output.DICT, timeout=self.timeout)
        except pt.TesseractNotFoundError:
            return failed("tesseract_not_found")
        except pt.TesseractError:
            logger.warning("tesseract error (lang=%s)", self.lang)
            return failed("tesseract_error")
        except RuntimeError:
            # pytesseract raises a bare RuntimeError on timeout
            return failed("tesseract_timeout")
        except Exception:
            logger.exception("tesseract OCR failed")
            return failed("ocr_failed")

        result = parse_image_to_data(data, language=self.lang)
        result.text_coverage = ink_text_coverage(rgb, result.words)
        result.processing_time_ms = (time.perf_counter() - t0) * 1000
        return result

    def version(self) -> str | None:
        return _tesseract_version(self.cmd)

    def languages(self) -> list[str]:
        try:
            return list(self._pytesseract().get_languages(config=""))
        except Exception:
            return []

    def status(self) -> dict[str, Any]:
        version = self.version()
        langs = self.languages() if version else []
        wanted = [p for p in self.lang.split("+") if p]
        lang_ok = bool(version) and all(p in langs for p in wanted)
        return {
            "available": bool(version),
            "version": version,
            "lang": self.lang,
            "lang_available": lang_ok,
        }


def _binary_stamp(cmd: str) -> int:
    """mtime of the resolved binary (0 if missing), so installs/upgrades invalidate the version cache."""
    path = shutil.which(cmd or "tesseract")
    try:
        return os.stat(path).st_mtime_ns if path else 0
    except OSError:
        return 0


def _tesseract_version(cmd: str) -> str | None:
    return _tesseract_version_at(cmd, _binary_stamp(cmd))


@functools.lru_cache(maxsize=8)
def _tesseract_version_at(cmd: str, _stamp: int) -> str | None:
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd or "tesseract"
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None

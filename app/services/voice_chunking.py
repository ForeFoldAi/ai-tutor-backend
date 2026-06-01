"""
Responsive text chunking for low-latency voice TTS enqueueing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

_IDLE_FLUSH_SEC = 0.5
_MIN_WORDS = 15
_MIN_CHARS = 60
_SOFT_PUNCT_MIN_CHARS = 40


def _word_count(text: str) -> int:
    return len(text.split())


def extract_responsive_chunks(buf: str) -> tuple[list[str], str]:
    """
    Extract speakable chunks from *buf* when any trigger matches:
      - sentence punctuation (.?!)
      - >= 15 words
      - >= 60 characters
      - comma/semicolon after 40 characters
    """
    out: list[str] = []

    while buf.strip():
        progressed = False

        parts = re.split(r"(?<=[.?!])\s+", buf)
        if len(parts) > 1:
            for part in parts[:-1]:
                chunk = part.strip()
                if chunk:
                    out.append(chunk)
            buf = parts[-1]
            progressed = True
            continue

        if _word_count(buf) >= _MIN_WORDS:
            words = buf.split()
            out.append(" ".join(words[:_MIN_WORDS]))
            buf = " ".join(words[_MIN_WORDS:])
            progressed = True
            continue

        if len(buf) >= _MIN_CHARS:
            flushed = False
            for sep in (", ", "; ", ": "):
                pos = buf.rfind(sep, _SOFT_PUNCT_MIN_CHARS)
                if pos >= _SOFT_PUNCT_MIN_CHARS:
                    out.append(buf[: pos + len(sep)].strip())
                    buf = buf[pos + len(sep) :].lstrip()
                    flushed = True
                    break
            if not flushed:
                pos = buf.rfind(" ", _SOFT_PUNCT_MIN_CHARS)
                if pos > 0:
                    out.append(buf[:pos].strip())
                    buf = buf[pos:].lstrip()
                else:
                    out.append(buf.strip())
                    buf = ""
            progressed = True
            continue

        if len(buf) >= _SOFT_PUNCT_MIN_CHARS:
            for sep in (", ", "; "):
                pos = buf.rfind(sep, 20)
                if pos >= 20:
                    out.append(buf[: pos + len(sep)].strip())
                    buf = buf[pos + len(sep) :].lstrip()
                    progressed = True
                    break
            if progressed:
                continue

        break

    return out, buf


@dataclass
class VoicePipelineTiming:
    """Per-turn latency markers (monotonic clock)."""

    turn_id: str = ""
    _t0: float = field(default_factory=time.monotonic)
    llm_first_token_at: float | None = None
    first_sentence_queued_at: float | None = None
    tts_started_at: float | None = None
    first_mp3_generated_at: float | None = None
    first_chunk_sent_at: float | None = None
    sentences_queued: int = 0

    def _ms(self, t: float | None) -> str:
        if t is None:
            return "—"
        return f"{(t - self._t0) * 1000:.0f}ms"

    def mark_llm_first_token(self) -> None:
        if self.llm_first_token_at is None:
            self.llm_first_token_at = time.monotonic()
            self._log("LLM first token", self.llm_first_token_at)

    def mark_sentence_queued(self, text: str) -> None:
        now = time.monotonic()
        self.sentences_queued += 1
        if self.first_sentence_queued_at is None:
            self.first_sentence_queued_at = now
            self._log("Sentence queued", now, preview=text[:48])

    def mark_tts_started(self, text: str) -> None:
        if self.tts_started_at is None:
            self.tts_started_at = time.monotonic()
            self._log("TTS started", self.tts_started_at, preview=text[:48])

    def mark_first_mp3_generated(self) -> None:
        if self.first_mp3_generated_at is None:
            self.first_mp3_generated_at = time.monotonic()
            self._log("First MP3 chunk generated", self.first_mp3_generated_at)

    def mark_first_chunk_sent(self) -> None:
        if self.first_chunk_sent_at is None:
            self.first_chunk_sent_at = time.monotonic()
            self._log("First MP3 chunk sent", self.first_chunk_sent_at)

    def summary(self) -> dict[str, str]:
        return {
            "llm_first_token": self._ms(self.llm_first_token_at),
            "sentence_queued": self._ms(self.first_sentence_queued_at),
            "tts_started": self._ms(self.tts_started_at),
            "first_mp3": self._ms(self.first_mp3_generated_at),
            "first_sent": self._ms(self.first_chunk_sent_at),
            "sentences_queued": str(self.sentences_queued),
        }

    def _log(self, label: str, t: float, **extra: str) -> None:
        import logging

        extra_s = " ".join(f"{k}={v!r}" for k, v in extra.items())
        logging.getLogger("voice.timing").info(
            "[voice %s] %s @ %s %s",
            self.turn_id or "?",
            label,
            self._ms(t),
            extra_s,
        )

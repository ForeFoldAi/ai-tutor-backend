"""
Speech-unit chunking for low-latency, continuous-sounding voice TTS.

Prefer natural pauses at periods and commas; strong discourse markers next.
First unit is aggressive for sub-600ms first-audio; later units grow for prosody.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.config import (
    VOICE_IDLE_FLUSH_SEC,
    VOICE_IDLE_FLUSH_STEADY_SEC,
    VOICE_SPEECH_FIRST_CHARS,
    VOICE_SPEECH_FIRST_WORDS,
    VOICE_SPEECH_MAX_WORDS,
    VOICE_SPEECH_STEADY_CHARS,
    VOICE_SPEECH_STEADY_WORDS,
)

# Legacy defaults (text chat / non-voice)
_IDLE_FLUSH_SEC = 0.5
_MIN_WORDS = 15
_MIN_CHARS = 60
_SOFT_PUNCT_MIN_CHARS = 40

# Clause / breath boundaries — only used when a unit already hit max_words.
_CLAUSE_SEPS = (", ", " — ", " - ", "; ", ": ")
# Forced overflow only — do not split mid-thought on because/so/but.
_DISCOURSE_SEPS = (
    " because ",
    " so ",
    " but ",
    " when ",
    " while ",
    " which ",
    " then ",
    " also ",
)
# Semantic teaching turns — new TTS unit starts at the marker (example / recap).
_TEACHING_SEPS = (
    ". First, ",
    ". Next, ",
    ". Finally, ",
    ". For example, ",
    ". The important ",
    ". Remember, ",
    ". Let's ",
    ". So, ",
    " — for example, ",
)
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
# Don't treat these periods as sentence ends (Dr. / e.g. / initials / 10 a.m.).
_ABBREV_END = re.compile(
    r"(?:"
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Fig|Ch|No|Vol|Rs|St|approx|Govt)\."
    r"|\b(?:e\.g|i\.e|a\.m|p\.m)\."
    r"|\b[A-Z]\."
    r")$",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _find_best_split(
    buf: str,
    *,
    min_chars: int,
    target_words: int,
    max_words: int,
    lookahead_chars: int = 0,
) -> int | None:
    """Return split index (end of left chunk) or None.

    Priority: sentence end (.?!) → teaching marker → (overflow only) comma/discourse → word.
    look-ahead: prefer not cutting when only a tiny remainder exists and more
    text is still streaming — reduces mid-thought TTS gaps.
    """
    n = len(buf)
    if n < min_chars and _word_count(buf) < target_words:
        return None

    def _ok_remainder(end: int) -> bool:
        if lookahead_chars <= 0:
            return True
        rem = n - end
        if rem >= lookahead_chars:
            return True
        # Allow short remainder only when we must flush (hit max_words)
        return _word_count(buf) >= max_words

    # 1) Sentence end — periods / ? / ! (skip Dr. / e.g. / initials)
    acc = ""
    last = 0
    for m in _SENTENCE_END.finditer(buf):
        left = buf[last : m.start()]
        piece = (acc + left).strip() if acc else left.strip()
        if not piece:
            acc = buf[last : m.end()]
            last = m.end()
            continue
        if _ABBREV_END.search(left.rstrip() or piece):
            acc = (acc + buf[last : m.end()]) if acc else buf[last : m.end()]
            last = m.end()
            continue
        wc = _word_count(piece)
        end = m.end()
        if (
            wc >= max(1, min(target_words, 3))
            and wc <= max_words
            and len(piece) >= min(min_chars, 8)
            and _ok_remainder(end)
        ):
            return end
        acc = buf[:end]
        last = m.end()

    # 2) Teaching transitions (example / recap) — new semantic unit
    best_teaching = -1
    for sep in _TEACHING_SEPS:
        pos = buf.rfind(sep, min_chars)
        if pos >= min_chars:
            end = pos + 1  # split after the period, keep marker with the next unit
            if end < min_chars:
                continue
            if _word_count(buf[:end]) <= max_words and _ok_remainder(end):
                best_teaching = max(best_teaching, end)
    if best_teaching > 0:
        return best_teaching

    # 3–4) Comma / discourse — only when we must flush (hit max_words)
    if _word_count(buf) >= max_words:
        for sep in _CLAUSE_SEPS:
            pos = buf.rfind(sep, min_chars)
            if pos >= min_chars:
                end = pos + len(sep)
                if _word_count(buf[:end]) <= max_words:
                    return end
        best = -1
        for sep in _DISCOURSE_SEPS:
            pos = buf.rfind(sep, min_chars)
            if pos >= min_chars:
                end = pos + len(sep)
                if _word_count(buf[:end]) <= max_words:
                    best = max(best, end)
        if best > 0:
            return best

    # 5) Word boundary fallback
    if _word_count(buf) >= max_words:
        words = buf.split()
        left = " ".join(words[:max_words])
        return len(left)

    if n >= min_chars + 8:
        pos = buf.rfind(" ", min_chars)
        if pos > min_chars and _ok_remainder(pos):
            return pos

    return None


def _thresholds(chunks_emitted: int) -> tuple[int, int, int, int]:
    """(min_words, min_chars, target_words, max_words) — adaptive per turn."""
    if chunks_emitted == 0:
        return (
            VOICE_SPEECH_FIRST_WORDS,
            VOICE_SPEECH_FIRST_CHARS,
            VOICE_SPEECH_STEADY_WORDS,
            VOICE_SPEECH_MAX_WORDS,
        )
    return (
        VOICE_SPEECH_STEADY_WORDS,
        VOICE_SPEECH_STEADY_CHARS,
        VOICE_SPEECH_STEADY_WORDS,
        VOICE_SPEECH_MAX_WORDS,
    )


def _extract_speech_units(buf: str, *, chunks_emitted: int = 0) -> tuple[list[str], str]:
    from app.config import VOICE_TTS_LOOKAHEAD_CHARS

    out: list[str] = []
    min_w, min_c, target_w, max_w = _thresholds(chunks_emitted)
    lookahead = VOICE_TTS_LOOKAHEAD_CHARS if chunks_emitted > 0 else 0

    while buf.strip():
        split_at = _find_best_split(
            buf,
            min_chars=min_c,
            target_words=target_w,
            max_words=max_w,
            lookahead_chars=lookahead,
        )
        if split_at is None:
            break
        chunk = buf[:split_at].strip()
        buf = buf[split_at:].lstrip()
        if chunk:
            out.append(chunk)
            chunks_emitted += 1
            min_w, min_c, target_w, max_w = _thresholds(chunks_emitted)
            lookahead = VOICE_TTS_LOOKAHEAD_CHARS

    return out, buf


def _extract_chunks(
    buf: str,
    *,
    min_words: int,
    min_chars: int,
    soft_punct_min: int,
) -> tuple[list[str], str]:
    """Legacy text-chat chunking."""
    out: list[str] = []
    while buf.strip():
        parts = _SENTENCE_END.split(buf)
        if len(parts) > 1:
            for part in parts[:-1]:
                chunk = part.strip()
                if chunk:
                    out.append(chunk)
            buf = parts[-1]
            continue
        words = buf.split()
        if len(words) >= min_words:
            out.append(" ".join(words[:min_words]))
            buf = " ".join(words[min_words:])
            continue
        if len(buf) >= min_chars:
            flushed = False
            for sep in _CLAUSE_SEPS:
                pos = buf.rfind(sep, soft_punct_min)
                if pos >= soft_punct_min:
                    out.append(buf[: pos + len(sep)].strip())
                    buf = buf[pos + len(sep) :].lstrip()
                    flushed = True
                    break
            if not flushed:
                pos = buf.rfind(" ", soft_punct_min)
                if pos > 0:
                    out.append(buf[:pos].strip())
                    buf = buf[pos:].lstrip()
                else:
                    out.append(buf.strip())
                    buf = ""
            continue
        if len(buf) >= soft_punct_min:
            for sep in (", ", "; "):
                pos = buf.rfind(sep, 20)
                if pos >= 20:
                    out.append(buf[: pos + len(sep)].strip())
                    buf = buf[pos + len(sep) :].lstrip()
                    break
            else:
                break
            continue
        break
    return out, buf


def extract_responsive_chunks(buf: str) -> tuple[list[str], str]:
    return _extract_chunks(buf, min_words=_MIN_WORDS, min_chars=_MIN_CHARS, soft_punct_min=_SOFT_PUNCT_MIN_CHARS)


def has_unclosed_math_delimiters(buf: str) -> bool:
    i = 0
    n = len(buf)
    while i < n:
        if buf.startswith("$$", i):
            close = buf.find("$$", i + 2)
            if close == -1:
                return True
            i = close + 2
            continue
        if buf[i] == "$":
            close = buf.find("$", i + 1)
            if close == -1:
                return True
            i = close + 1
            continue
        i += 1
    return False


def extract_voice_chunks(buf: str, *, chunks_emitted: int = 0) -> tuple[list[str], str]:
    """Speech-unit extraction for live voice (not sentence-per-TTS)."""
    if has_unclosed_math_delimiters(buf):
        return [], buf
    return _extract_speech_units(buf, chunks_emitted=chunks_emitted)


def idle_flush_sec(*, chunks_emitted: int) -> float:
    """Shorter idle flush for first unit; slightly longer once speech is flowing."""
    if chunks_emitted == 0:
        return VOICE_IDLE_FLUSH_SEC
    return VOICE_IDLE_FLUSH_STEADY_SEC


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
    tts_prefetch_hits: int = 0
    tts_prefetch_misses: int = 0
    tts_units_played: int = 0
    tts_queue_high_water: int = 0
    tokens_streamed: int = 0
    queue_depth: int = 0
    rag_completed_at: float | None = None
    _last_playback_gap_ms: float = 0.0

    def mark_rag_done(self) -> None:
        if self.rag_completed_at is None:
            self.rag_completed_at = time.monotonic()
            self._log("RAG complete", self.rag_completed_at)

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
        self.tts_queue_high_water = max(self.tts_queue_high_water, self.sentences_queued)
        if self.first_sentence_queued_at is None:
            self.first_sentence_queued_at = now
            self._log("Speech unit queued", now, preview=text[:48])

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

    def mark_prefetch_hit(self) -> None:
        self.tts_prefetch_hits += 1

    def mark_prefetch_miss(self) -> None:
        self.tts_prefetch_misses += 1

    def mark_tts_unit_played(self) -> None:
        self.tts_units_played += 1

    def mark_playback_gap(self, gap_ms: float) -> None:
        """Record inter-unit playback gap for continuity metrics."""
        self._last_playback_gap_ms = gap_ms

    def record_tts_units_played(self, count: int) -> None:
        self.tts_units_played = count

    def record_queue_depth(self, depth: int) -> None:
        self.queue_depth = depth
        self.tts_queue_high_water = max(self.tts_queue_high_water, depth)

    def record_tokens_streamed(self, count: int) -> None:
        self.tokens_streamed = count

    def summary(self) -> dict[str, str]:
        return {
            "llm_first_token": self._ms(self.llm_first_token_at),
            "speech_unit_queued": self._ms(self.first_sentence_queued_at),
            "tts_started": self._ms(self.tts_started_at),
            "first_mp3": self._ms(self.first_mp3_generated_at),
            "first_sent": self._ms(self.first_chunk_sent_at),
            "speech_units": str(self.sentences_queued),
            "tokens_streamed": str(self.tokens_streamed),
            "queue_depth": str(self.queue_depth),
            "prefetch_hits": str(self.tts_prefetch_hits),
            "prefetch_misses": str(self.tts_prefetch_misses),
            "tts_units_played": str(self.tts_units_played),
            "queue_high_water": str(self.tts_queue_high_water),
            "rag_complete": self._ms(self.rag_completed_at),
            "last_playback_gap_ms": f"{self._last_playback_gap_ms:.0f}",
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

"""
Chapter vocabulary for speech-to-text accuracy.

Speech recognisers confuse homophones ("Tricki" the dog vs "tricky", "vet" vs
"wet"). The textbook chapter tells us which word the student most likely meant,
so we (1) bias Whisper with chapter terms, (2) snap near-homophones to frequent
chapter terms, and (3) give the LLM a short "possible mishears" hint.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")

# Function / very common words are never "corrected" (what != vet).
_COMMON = frozenset(
    """a about after all also am an and any are as at be because been before but by can could did do does
    done for from get give go had has have he her here him his how i if in into is it its just like make me
    more most my no not now of on one or our out over say she should so some than that the their them then
    there these they this to up us was we were what when where which who whom why will with would you your
    yes okay ok tell explain story chapter question answer want know mean said say good""".split()
)

_PHON_GROUPS = {
    "v": "f", "w": "f", "f": "f", "b": "p", "p": "p", "d": "t", "t": "t",
    "g": "k", "k": "k", "c": "k", "q": "k", "s": "s", "z": "s", "m": "n", "n": "n",
    "y": "i",
}


def phonetic_key(word: str) -> str:
    """Coarse consonant skeleton: confusable consonants merge, vowels/h dropped."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    out: list[str] = []
    for i, ch in enumerate(w):
        ch = _PHON_GROUPS.get(ch, ch)
        if ch in "aeiouh" and i > 0:
            continue
        if ch == "i" and i > 0:  # y/i behave like vowels past the first letter
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


@dataclass
class ChapterVocab:
    counts: Counter = field(default_factory=Counter)  # lowercase word -> chapter frequency
    proper: dict[str, str] = field(default_factory=dict)  # lowercase -> canonical casing
    key_terms: list[str] = field(default_factory=list)
    _by_phon: dict[str, list[str]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.counts

    def _index(self) -> None:
        by: dict[str, list[str]] = {}
        for w, n in self.counts.items():
            if len(w) < 3 or w in _COMMON or (n < 2 and w not in self.proper):
                continue
            by.setdefault(phonetic_key(w), []).append(w)
        self._by_phon = by

    def whisper_prompt_terms(self, limit: int = 25) -> str:
        terms = list(self.proper.values())[:limit]
        for t in self.key_terms:
            if len(terms) >= limit:
                break
            if t not in terms:
                terms.append(t)
        return ", ".join(terms)

    def relevance(self, text: str) -> float:
        """Share of the student's content words that appear in the chapter (0..1)."""
        words = [w.lower() for w in _WORD.findall(text or "") if w.lower() not in _COMMON and len(w) > 2]
        if not words:
            return 1.0  # nothing to judge -> do not penalise
        hit = sum(1 for w in words if self.counts.get(w, 0) > 0 or self._stem_hit(w))
        return hit / len(words)

    def _stem_hit(self, w: str) -> bool:
        return len(w) >= 5 and any(c.startswith(w[:5]) for c in self.counts if len(c) >= 5)

    def resolve(self, text: str) -> tuple[str, list[str]]:
        """Return (corrected_text, hints).

        A word is replaced when a much more frequent chapter term sounds like it
        and the word itself is rare in the chapter. Ambiguous cases only produce
        a hint for the LLM instead of an edit.
        """
        if self.empty or not text:
            return text, []
        if not self._by_phon:
            self._index()
        hints: list[str] = []

        def repl(m: re.Match) -> str:
            word = m.group(0)
            low = word.lower()
            if low in _COMMON or len(low) < 3 or low in self.proper:
                return word
            own = self.counts.get(low, 0)
            best: tuple[float, str] | None = None
            for cand in self._by_phon.get(phonetic_key(low), []):
                if cand == low or abs(len(cand) - len(low)) > 2:
                    continue
                ratio = difflib.SequenceMatcher(None, low, cand).ratio()
                if ratio < 0.6:
                    continue
                if best is None or ratio > best[0]:
                    best = (ratio, cand)
            if best is None:
                return word
            cand = best[1]
            cand_n = self.counts.get(cand, 0)
            canon = self.proper.get(cand, cand)
            if cand_n >= 3 * max(own, 1) and own <= 1:
                if word[:1].isupper() and word.islower() is False:
                    canon = canon[:1].upper() + canon[1:]
                return canon
            if cand_n > own:
                hints.append(f'"{word}" may be the chapter word "{canon}"')
            return word

        out = _WORD.sub(repl, text)
        return out, hints[:4]


def build_vocab_from_texts(texts: list[str], names: list[str] | None = None) -> ChapterVocab:
    counts: Counter = Counter()
    caps: Counter = Counter()
    for t in texts:
        # sentence-initial capitals are not proper nouns: only count mid-sentence capitals
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", t):
            caps[m.group(1)] += 1
        for w in _WORD.findall(t):
            counts[w.lower()] += 1
    proper: dict[str, str] = {}
    for w, n in caps.most_common(40):
        low = w.lower()
        if n >= 2 and low not in _COMMON:
            proper[low] = w
    for name in names or []:
        for w in _WORD.findall(name):
            if len(w) > 3 and w.lower() not in _COMMON:
                counts[w.lower()] += 2
    key_terms = [w for w, n in counts.most_common(300) if len(w) >= 5 and w not in _COMMON and w not in proper][:20]
    v = ChapterVocab(counts=counts, proper=proper, key_terms=key_terms)
    v._index()
    return v


def load_chapter_vocab(collection: str, chapter_ids: list[str] | None, chapter_names: list[str] | None) -> ChapterVocab:
    """Blocking: fetch chapter chunks and build the vocabulary. Run in a thread."""
    if not collection or not chapter_ids:
        return build_vocab_from_texts([], chapter_names)
    try:
        from app.services.vector_service import fetch_chapter_chunks

        docs = fetch_chapter_chunks(collection, list(chapter_ids), chapter_names=chapter_names, limit=400)
        texts = [getattr(d, "page_content", "") or "" for d in docs]
        return build_vocab_from_texts(texts, chapter_names)
    except Exception as exc:  # never block a session on vocabulary
        logger.debug("chapter vocab load failed: %s", exc)
        return build_vocab_from_texts([], chapter_names)


# ── Student correction detection ("I didn't say that", "no I meant X") ──────────
_CORRECTION = re.compile(
    r"\b(i\s+(?:am|was|did)\s*n[o']?t\s+(?:say|ask)(?:ing)?|i\s+didn'?t\s+say|"
    r"i\s+(?:said|meant|mean|am\s+saying|was\s+saying)|that'?s\s+not\s+what\s+i\s+(?:said|asked)|"
    r"not\s+what\s+i\s+asked|wrong\s+(?:word|question)|no\s*,?\s+i\s+(?:said|meant))\b",
    re.I,
)


def is_student_correction(text: str) -> bool:
    return bool(_CORRECTION.search(text or ""))


# ── Chapter overview intent ("explain the chapter from start to end") ──────────────
_OVERVIEW = re.compile(
    r"\b(explain|tell|teach|summari[sz]e|walk\s+me\s+through|go\s+through)\b.{0,40}"
    r"\b(whole|entire|full|complete|this)?\s*(chapter|story|lesson|topic)\b|"
    r"\bstart(?:ing)?\s+(?:to|till|until)\s+(?:the\s+)?end\b|\bfrom\s+(?:the\s+)?(?:start|beginning)\b|"
    r"\bwhat\s+is\s+(?:this|the)\s+(?:chapter|story|lesson)\s+(?:all\s+)?about\b",
    re.I,
)


def is_chapter_overview_request(text: str) -> bool:
    return bool(_OVERVIEW.search(text or ""))

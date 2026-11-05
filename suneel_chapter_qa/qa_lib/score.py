"""Heuristic turn scoring (keywords, forbidden content, images, interactive panels)."""
from __future__ import annotations

# Channels that never render math / science panels.
_NO_PANEL_CHANNELS = frozenset({"voice", "assistant"})


def score_turn(
    *,
    answer: str,
    images: list | None,
    math_lesson,
    science_experiment,
    expect_keywords: list[str],
    forbidden: list[str],
    label: str,
    want_images: bool,
    want_math: bool,
    want_science: bool,
    channel: str,
) -> dict:
    a = (answer or "").lower()
    bugs: list[str] = []

    ans = 5.0
    if len(a) < 40:
        ans = 2.0
        bugs.append("very short answer")
    elif len(a) > 80:
        ans = 7.0
    hits = sum(1 for k in expect_keywords if k.lower() in a)
    if hits:
        ans = min(10.0, ans + 1.5 * hits)
    else:
        ans = max(1.0, ans - 3.0)
        bugs.append("missing expected keywords")
    for f in forbidden:
        if f.lower() in a:
            ans = max(1.0, ans - 2.5)
            bugs.append(f"forbidden content mentioned: {f}")
    if label in ("wrong", "typo") and any(
        w in a for w in ("not", "doesn't", "does not", "isn't", "chapter")
    ):
        ans = min(10.0, ans + 1.0)

    denom = max(1, len(expect_keywords))
    expect = round(10.0 * hits / denom, 1)

    nimg = len(images or [])
    if want_images and channel != "assistant":
        if nimg >= 1:
            images_score = 9.0
        else:
            images_score = 2.0
            bugs.append("wanted images but got none")
    else:
        images_score = 9.0 if nimg else 8.0

    interactive: float | None
    if channel in _NO_PANEL_CHANNELS:
        interactive = None
        if want_math or want_science:
            if len(a) > 100:
                interactive_note = f"N/A ({channel}); transcript length ok"
            else:
                interactive_note = f"N/A ({channel}); thin transcript"
                bugs.append(f"{channel} transcript thin for interactive topic")
        else:
            interactive_note = "N/A"
    else:
        interactive_note = ""
        if want_math:
            if math_lesson:
                interactive = 9.0
            else:
                interactive = 3.0
                bugs.append("expected math_lesson missing")
        elif want_science:
            if science_experiment:
                interactive = 9.0
            else:
                interactive = 4.0
                bugs.append("expected science_experiment missing")
        else:
            interactive = 8.0 if not math_lesson and not science_experiment else 7.0
            if math_lesson or science_experiment:
                interactive_note = "unexpected interactive panel"

    parts = [ans, expect, images_score]
    if interactive is not None:
        parts.append(interactive)
    overall = round(sum(parts) / len(parts), 1)

    return {
        "answer_score": round(ans, 1),
        "expect_score": expect,
        "images_score": round(images_score, 1),
        "interactive_score": interactive,
        "interactive_note": interactive_note,
        "overall": overall,
        "bugs": bugs,
        "keyword_hits": hits,
    }

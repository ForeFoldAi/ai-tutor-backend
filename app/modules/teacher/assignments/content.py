"""Normalize lesson artifacts into student_content + answer_key; grade submissions."""

from __future__ import annotations

import re
from typing import Any

_ANSWER_SECTION = re.compile(
    r"(?im)^#{1,3}\s*(answer\s*key|answers?|solutions?)\s*$.*?(?=^#{1,3}\s|\Z)",
    re.DOTALL,
)
_OPTION_LINE = re.compile(r"(?m)^\s*([A-D])[\.\)\-]\s+(.+?)\s*$")
# Allow Answer: B, Answer: **B**, **Answer:** B, Correct answer - A
_INLINE_ANSWER = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:Answer|Correct(?:\s*answer)?)\s*(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*([A-D])\b"
)
_ANSWER_KEY_LINE = re.compile(
    r"(?m)^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?(\d+)[\.\)\-]\s*(?:\*\*)?\s*([A-D])\b"
)


def _norm_letter(value: Any) -> str:
    s = str(value or "").strip().upper().replace("*", "").strip()
    if not s:
        return ""
    m = re.search(r"[A-D]", s)
    return m.group(0) if m else s


def _as_dict(content: Any) -> dict[str, Any]:
    return content if isinstance(content, dict) else {}


def _markdown(content: dict[str, Any]) -> str | None:
    if content.get("format") == "markdown" and isinstance(content.get("markdown"), str):
        return content["markdown"]
    if isinstance(content.get("markdown"), str) and not any(
        k in content for k in ("mcq", "fill_blanks", "practice_questions", "true_false")
    ):
        return content["markdown"]
    return None


def _strip_answers_md(md: str) -> str:
    return _ANSWER_SECTION.sub("", md).strip()


def _parse_answer_key_section(md: str) -> dict[str, str]:
    """Parse '## Answer Key' / '## Answers' blocks into qN → letter."""
    section = _ANSWER_SECTION.search(md)
    if not section:
        return {}
    out: dict[str, str] = {}
    for m in _ANSWER_KEY_LINE.finditer(section.group(0)):
        out[f"q{m.group(1)}"] = m.group(2).upper()
    return out


def _parse_mcq_from_markdown(md: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Best-effort MCQ parse; ceiling: non-standard markdown formats."""
    questions: list[dict[str, Any]] = []
    answers: dict[str, str] = {}
    # Prefer dedicated answer-key section, then fill from inline Answer: lines.
    answers.update(_parse_answer_key_section(md))
    body = _strip_answers_md(md)
    # Split on numbered questions
    parts = re.split(r"(?m)(?=^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?\d+[\.\)])", body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(
            r"(?ms)^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?(\d+)[\.\)]\s*\*?\*?\s*(.+?)(?=\n\s*[A-D][\.\)\-]|\Z)",
            part,
        )
        if not m:
            continue
        qid = m.group(1)
        stem = m.group(2).strip()
        options = [
            {"key": om.group(1).upper(), "text": om.group(2).strip()}
            for om in _OPTION_LINE.finditer(part)
        ]
        if len(options) < 2:
            continue
        ans_m = _INLINE_ANSWER.search(part)
        item_id = f"q{qid}"
        questions.append(
            {
                "id": item_id,
                "number": int(qid),
                "question": stem,
                "type": "mcq",
                "options": [o["text"] for o in options],
                "option_keys": [o["key"] for o in options],
            }
        )
        if item_id not in answers and ans_m:
            answers[item_id] = ans_m.group(1).upper()
    # If we only found answers from the key section but questions were parsed
    # from a body that still included inline answers, merge any leftover inlines
    # from the original markdown (before strip) for questions we already have.
    if questions:
        for part in re.split(r"(?m)(?=^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?\d+[\.\)])", md):
            ans_m = _INLINE_ANSWER.search(part or "")
            qm = re.match(
                r"(?ms)^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?(\d+)[\.\)]",
                (part or "").strip(),
            )
            if not qm or not ans_m:
                continue
            item_id = f"q{qm.group(1)}"
            if item_id not in answers:
                answers[item_id] = ans_m.group(1).upper()
    return questions, answers


def _clean_short_stem(stem: str) -> str:
    """Drop answer blanks and trailing section dividers glued onto a question."""
    text = stem.strip()
    text = re.sub(r"(?im)^\s*Answer\s*:?\s*_+\s*$", "", text)
    text = re.sub(r"(?im)\n?\s*Answer\s*:?\s*_+\s*", "\n", text)
    text = re.sub(r"(?ms)\n(?:---\s*\n)?##\s+[^\n]+\s*$", "", text)
    text = re.sub(r"(?ms)\n---\s*$", "", text)
    return text.strip()


def _parse_numbered_short_from_markdown(md: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Parse 1. / 2. open-ended worksheet (or quiz) questions when MCQ options are absent."""
    questions: list[dict[str, Any]] = []
    answers: dict[str, str] = {}
    answers.update(_parse_answer_key_section(md))
    body = _strip_answers_md(md)
    parts = re.split(r"(?m)(?=^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?\d+[\.\)]\s+)", body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(
            r"(?ms)^\s*(?:\*\*)?(?:Q(?:uestion)?\s*)?(\d+)[\.\)]\s*\*?\*?\s*(.+)",
            part,
        )
        if not m:
            continue
        qid = m.group(1)
        stem = _clean_short_stem(m.group(2))
        if len(stem) < 8:
            continue
        # Skip preamble mistaken as Q1 when stem is mostly title/meta
        if _HW_HEADING.match(stem) or (stem.lstrip().startswith("#") and len(stem) < 80):
            continue
        options = [
            {"key": om.group(1).upper(), "text": om.group(2).strip()}
            for om in _OPTION_LINE.finditer(part)
        ]
        item_id = f"q{qid}"
        if len(options) >= 2:
            questions.append(
                {
                    "id": item_id,
                    "number": int(qid),
                    "question": re.split(r"(?m)^\s*[A-D][\.\)\-]", stem)[0].strip() or stem,
                    "type": "mcq",
                    "options": [o["text"] for o in options],
                    "option_keys": [o["key"] for o in options],
                }
            )
        else:
            questions.append(
                {
                    "id": item_id,
                    "number": int(qid),
                    "question": stem[:4000],
                    "type": "short",
                }
            )
        ans_m = _INLINE_ANSWER.search(part)
        if item_id not in answers and ans_m:
            answers[item_id] = ans_m.group(1).upper()
    return questions, answers


def _structured_quiz(content: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    questions: list[dict[str, Any]] = []
    answers: dict[str, str] = {}
    raw_answers = content.get("answers") if isinstance(content.get("answers"), dict) else {}

    for i, q in enumerate(content.get("mcq") or []):
        if not isinstance(q, dict):
            continue
        item_id = f"mcq_{i + 1}"
        opts = q.get("options") or []
        if isinstance(opts, dict):
            option_keys = [str(k).upper() for k in opts.keys()]
            option_texts = [str(v) for v in opts.values()]
        else:
            option_texts = [str(o) for o in opts]
            option_keys = [chr(65 + j) for j in range(len(option_texts))]
        questions.append(
            {
                "id": item_id,
                "number": i + 1,
                "question": str(q.get("question") or ""),
                "type": "mcq",
                "options": option_texts,
                "option_keys": option_keys,
            }
        )
        ans = (
            q.get("answer")
            or q.get("correct")
            or q.get("correct_answer")
            or raw_answers.get(item_id)
            or raw_answers.get(str(i + 1))
        )
        if ans is not None:
            answers[item_id] = _norm_letter(ans)

    offset = len(questions)
    for i, q in enumerate(content.get("short_answer") or []):
        if not isinstance(q, dict):
            continue
        item_id = f"short_{i + 1}"
        questions.append(
            {
                "id": item_id,
                "number": offset + i + 1,
                "question": str(q.get("question") or ""),
                "type": "short",
            }
        )
        ans = q.get("answer") or raw_answers.get(item_id)
        if ans is not None:
            answers[item_id] = str(ans).strip()
    return questions, answers


def _structured_worksheet(content: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    questions: list[dict[str, Any]] = []
    answers: dict[str, str] = {}
    n = 0
    sections = (
        "fill_blanks",
        "true_false",
        "match_following",
        "short_answer",
        "long_answer",
        "application_questions",
    )
    for section in sections:
        for item in content.get(section) or []:
            if not isinstance(item, dict):
                continue
            n += 1
            item_id = f"{section}_{n}"
            if section == "true_false":
                qtext = str(item.get("statement") or item.get("question") or "")
                questions.append(
                    {
                        "id": item_id,
                        "number": n,
                        "question": qtext,
                        "type": "mcq",
                        "options": ["True", "False"],
                        "option_keys": ["A", "B"],
                    }
                )
                ans = item.get("answer") or item.get("correct")
                if ans is not None:
                    truth = str(ans).strip().lower() in ("true", "t", "yes", "1", "a")
                    answers[item_id] = "A" if truth else "B"
                continue
            if section == "match_following":
                qtext = f"Match: {item.get('left', '?')} → ?"
                questions.append({"id": item_id, "number": n, "question": qtext, "type": "short"})
                ans = item.get("answer") or item.get("correct") or item.get("right")
                if ans is not None:
                    answers[item_id] = str(ans).strip()
                continue
            qtext = str(item.get("question") or item.get("statement") or "")
            opts = item.get("options")
            if opts:
                if isinstance(opts, dict):
                    option_keys = [str(k).upper() for k in opts.keys()]
                    option_texts = [str(v) for v in opts.values()]
                else:
                    option_texts = [str(o) for o in opts]
                    option_keys = [chr(65 + j) for j in range(len(option_texts))]
                questions.append(
                    {
                        "id": item_id,
                        "number": n,
                        "question": qtext,
                        "type": "mcq",
                        "options": option_texts,
                        "option_keys": option_keys,
                    }
                )
            else:
                questions.append(
                    {"id": item_id, "number": n, "question": qtext, "type": "short"}
                )
            ans = item.get("answer") or item.get("correct") or item.get("blank")
            if ans is not None:
                answers[item_id] = str(ans).strip()
    return questions, answers


def _structured_homework(content: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tasks: list[dict[str, Any]] = []
    answers: dict[str, str] = {}
    sections = (
        "practice_questions",
        "observation_tasks",
        "project_work",
        "reading_assignment",
        "revision_tasks",
    )
    n = 0
    for section in sections:
        for item in content.get(section) or []:
            if not isinstance(item, dict):
                continue
            n += 1
            item_id = f"hw_{n}"
            tasks.append(
                {
                    "id": item_id,
                    "number": n,
                    "title": str(item.get("title") or item.get("task") or f"Task {n}"),
                    "description": str(item.get("description") or item.get("text") or item.get("question") or ""),
                    "type": "short",
                }
            )
            ans = item.get("answer")
            if ans is not None:
                answers[item_id] = str(ans).strip()
    return tasks, answers


_HW_HEADING = re.compile(r"^\s*#{1,6}\s+")
_HW_META = re.compile(
    r"(?i)^\s*(?:\*\*)?(class|grade|subject|chapter|board|curriculum|topic|title|name|date|school)\b"
)
_HW_SECTION = re.compile(
    r"(?i)^\s*(?:\*\*)?(practice|real[- ]?life|observation|creative|mini\s*projects?|"
    r"research|reflection)(?:\s+tasks?)?\s*(?:\*\*)?\s*$"
)
_HW_BULLET = re.compile(r"^\s*[-*•]\s+(.+)$")
_HW_NUMBERED = re.compile(r"^\s*\d+[\.\)]\s+(.+)$")
# "Science Timeline Challenge:** Create…" or "**Title:** Create…"
_HW_TITLED = re.compile(r"^\s*(?:\*\*)?(.+?)(?:\*\*)?:\*\*\s*(.+)$")
_HW_BOLD_TITLE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s+(.+)$")


def _hw_is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _HW_HEADING.match(s):
        return True
    bare = _HW_HEADING.sub("", s).strip(" *")
    if _HW_SECTION.match(bare) or _HW_SECTION.match(s.strip(" *")):
        return True
    if _HW_META.match(s) and len(s) < 160:
        return True
    return False


def _hw_split_titled(text: str) -> tuple[str, str] | None:
    m = _HW_TITLED.match(text) or _HW_BOLD_TITLE.match(text)
    if not m:
        return None
    title = m.group(1).strip(" *")
    body = m.group(2).strip()
    if not title or not body:
        return None
    if _HW_META.match(f"{title}:"):
        return None
    return title, body


def homework_items_need_repair(items: list[Any] | None) -> bool:
    """True when stored homework was built by the old line-per-task parser."""
    for it in items or []:
        if not isinstance(it, dict):
            continue
        desc = str(it.get("description") or it.get("question") or "")
        if _hw_is_noise(desc):
            return True
    return False


def worksheet_items_need_repair(items: list[Any] | None) -> bool:
    """True when worksheet was stored as one markdown blob instead of numbered items."""
    if not items or len(items) != 1:
        return False
    it = items[0]
    if not isinstance(it, dict):
        return False
    q = str(it.get("question") or it.get("description") or "")
    if len(q) < 200:
        return False
    if q.lstrip().startswith("#"):
        return True
    if "## " in q and re.search(r"\d+[\.\)]\s+\S+", q):
        return True
    return False


def _homework_from_markdown(md: str) -> list[dict[str, Any]]:
    """Parse homework markdown into actionable tasks (skip titles / section headers)."""
    clean = _strip_answers_md(md)
    tasks: list[dict[str, Any]] = []

    def add(title: str, body: str) -> None:
        body = body.strip()
        if len(body) < 20:
            return
        if _hw_is_noise(body) and not title:
            return
        n = len(tasks) + 1
        tasks.append(
            {
                "id": f"hw_{n}",
                "number": n,
                "title": title or f"Task {n}",
                "description": body,
                "type": "short",
            }
        )

    for raw in clean.splitlines():
        line = raw.strip()
        if not line:
            continue
        bullet = _HW_BULLET.match(line) or _HW_NUMBERED.match(line)
        if bullet:
            body = bullet.group(1).strip()
            titled = _hw_split_titled(body)
            if titled:
                add(titled[0], titled[1])
            elif not _hw_is_noise(body):
                add("", body)
            continue
        if _hw_is_noise(line):
            continue
        titled = _hw_split_titled(line)
        if titled:
            add(titled[0], titled[1])
            continue
        if len(line) >= 40:
            add("", line)

    if not tasks:
        kept: list[str] = []
        for ln in clean.splitlines():
            s = ln.strip()
            if not s or _hw_is_noise(s):
                continue
            bm = _HW_BULLET.match(s) or _HW_NUMBERED.match(s)
            kept.append(bm.group(1).strip() if bm else s.lstrip("-*• ").strip())
        # ponytail: one blob if structure is unknown; upgrade by teaching the generator a stable task schema
        blob = "\n".join(kept).strip()[:4000]
        tasks.append(
            {
                "id": "hw_1",
                "number": 1,
                "title": "Homework",
                "description": blob or "Complete the assigned homework.",
                "type": "short",
            }
        )
    return tasks


def normalize_artifact(
    artifact_type: str,
    content: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (student_content, answer_key)."""
    data = _as_dict(content)
    md = _markdown(data)

    if artifact_type == "quiz":
        if md:
            questions, answers = _parse_mcq_from_markdown(md)
            if not questions:
                questions, answers = _parse_numbered_short_from_markdown(md)
            if not questions:
                questions = [
                    {
                        "id": "q1",
                        "number": 1,
                        "question": _strip_answers_md(md)[:4000],
                        "type": "short",
                    }
                ]
            return {"kind": "quiz", "items": questions}, {"answers": answers}
        questions, answers = _structured_quiz(data)
        return {"kind": "quiz", "items": questions}, {"answers": answers}

    if artifact_type == "worksheet":
        if md:
            questions, answers = _parse_mcq_from_markdown(md)
            if not questions:
                questions, answers = _parse_numbered_short_from_markdown(md)
            if not questions:
                questions = [
                    {
                        "id": "w1",
                        "number": 1,
                        "question": _strip_answers_md(md)[:4000],
                        "type": "short",
                    }
                ]
            return {"kind": "worksheet", "items": questions}, {"answers": answers}
        questions, answers = _structured_worksheet(data)
        return {"kind": "worksheet", "items": questions}, {"answers": answers}

    # homework
    if md:
        return {"kind": "homework", "items": _homework_from_markdown(md)}, {"answers": {}}
    tasks, answers = _structured_homework(data)
    if not tasks:
        tasks = [
            {
                "id": "hw_1",
                "number": 1,
                "title": "Homework",
                "description": "Complete the assigned homework.",
                "type": "short",
            }
        ]
    return {"kind": "homework", "items": tasks}, {"answers": answers}


def _display_expected(item: dict[str, Any], expected: Any) -> str | None:
    if expected is None or str(expected).strip() == "":
        return None
    exp = str(expected).strip()
    if (item.get("type") or "") != "mcq":
        return exp
    letter = _norm_letter(exp)
    opts = item.get("options") or []
    keys = item.get("option_keys") or [chr(65 + i) for i in range(len(opts))]
    if letter in keys:
        idx = keys.index(letter)
        if 0 <= idx < len(opts):
            return f"{letter}. {opts[idx]}"
    return letter or exp


def grade_submission(
    student_content: dict[str, Any],
    answer_key: dict[str, Any],
    answers: dict[str, Any],
) -> tuple[float | None, float | None, dict[str, Any], str]:
    """Returns score, max_score, result detail, status (graded|submitted)."""
    items = student_content.get("items") or []
    key_answers = (answer_key or {}).get("answers") or {}
    details: list[dict[str, Any]] = []
    scored = 0
    correct = 0
    answered = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        raw = answers.get(item_id)
        student_ans = "" if raw is None else str(raw).strip()
        if student_ans:
            answered += 1
        expected = key_answers.get(item_id)
        qtype = item.get("type") or "short"
        entry: dict[str, Any] = {
            "id": item_id,
            "question": item.get("question") or item.get("title") or "",
            "student_answer": student_ans,
            "correct_answer": _display_expected(item, expected),
            "is_correct": None,
        }
        if expected is not None and str(expected).strip() != "":
            scored += 1
            if qtype == "mcq":
                ok = _norm_letter(student_ans) == _norm_letter(expected)
                # also accept option text match
                if not ok and student_ans:
                    opts = item.get("options") or []
                    keys = item.get("option_keys") or [chr(65 + i) for i in range(len(opts))]
                    exp_letter = _norm_letter(expected)
                    try:
                        idx = keys.index(exp_letter) if exp_letter in keys else -1
                    except ValueError:
                        idx = -1
                    if idx >= 0 and idx < len(opts) and student_ans.strip().lower() == str(opts[idx]).strip().lower():
                        ok = True
            else:
                ok = student_ans.strip().lower() == str(expected).strip().lower()
            entry["is_correct"] = ok
            if ok:
                correct += 1
        details.append(entry)

    if scored == 0:
        return None, None, {"items": details, "auto_scored": False}, "submitted"

    return float(correct), float(scored), {"items": details, "auto_scored": True}, "graded"


def _selfcheck() -> None:
    md = """
1. What is 2+2?
A. 3
B. 4
C. 5
D. 6
Answer: B

## Answer Key
1. B
"""
    content, key = normalize_artifact("quiz", {"format": "markdown", "markdown": md})
    assert content["kind"] == "quiz"
    assert len(content["items"]) >= 1
    assert key["answers"].get("q1") == "B"
    score, max_s, result, status = grade_submission(
        content, key, {"q1": "B"}
    )
    assert status == "graded" and score == 1 and max_s == 1
    assert result["items"][0]["is_correct"] is True
    assert "B." in (result["items"][0]["correct_answer"] or "")
    score2, _, _, _ = grade_submission(content, key, {"q1": "A"})
    assert score2 == 0

    bold = """
1. What is force?
A. Push
B. Pull
C. Mass
D. Speed
Answer: **A**

2. Unit?
A. Joule
B. Newton
C. Watt
D. Pascal
**Answer:** B

## Answer Key
1. A
2. B
"""
    c2, k2 = normalize_artifact("quiz", {"format": "markdown", "markdown": bold})
    assert k2["answers"].get("q1") == "A" and k2["answers"].get("q2") == "B"

    key_only = """
1. What is force?
A. Push
B. Pull
C. Mass
D. Speed

2. Unit?
A. Joule
B. Newton
C. Watt
D. Pascal

## Answer Key
1. A
2. B
"""
    c3, k3 = normalize_artifact("quiz", {"format": "markdown", "markdown": key_only})
    assert k3["answers"] == {"q1": "A", "q2": "B"}
    s3, m3, _, st3 = grade_submission(c3, k3, {"q1": "A", "q2": "C"})
    assert st3 == "graded" and s3 == 1 and m3 == 2

    hw = """
# Homework: The Ever-Evolving World of Science

Class:** 9
Subject:** Science
Chapter:** The Ever-Evolving World of Science

## Practice Tasks

Science Timeline Challenge:** Create a timeline of 10 major scientific discoveries or inventions. Include the year and impact.

Scientific Method Reflection:** Write a short paragraph explaining how the scientific method is used in everyday problem-solving.

## Real-Life Tasks

Household Science Hunt:** Identify 5 items in your home that are the result of scientific advancements.

## Observation Tasks

- Spend 15 minutes observing something connected to today's lesson. Record 5 observations in a notebook.
"""
    hw_content, _ = normalize_artifact("homework", {"format": "markdown", "markdown": hw})
    assert hw_content["kind"] == "homework"
    titles = [i["title"] for i in hw_content["items"]]
    descs = [i["description"] for i in hw_content["items"]]
    assert not any(d.lstrip().startswith("#") for d in descs)
    assert "Science Timeline Challenge" in titles
    assert "Scientific Method Reflection" in titles
    assert "Household Science Hunt" in titles
    assert any("observing something" in d.lower() for d in descs)
    assert homework_items_need_repair(
        [{"description": "# Homework: Science"}, {"description": "Class:** 9"}]
    )
    assert not homework_items_need_repair(hw_content["items"])

    ws = """
# Worksheet: The Ever-Evolving World of Science

**Class:** CLASS_9
**Subject:** Science
**Chapter:** 1 - The Ever-Evolving World of Science

---

## Warm-Up Questions

1. What does the term "science" mean to you? Write down three words that come to your mind when you think about science.

2. Why do you think science is important in our daily lives? Give one example.

---

## Practice Questions

3. Define the following terms in your own words:
a) Hypothesis
b) Observation
c) Experiment

4. Arrange the following steps of the scientific method in the correct order.

---

## Application Questions

5. Imagine you are conducting an experiment to test which type of soil is best for plant growth. Describe how you would set up this experiment.
"""
    ws_content, _ = normalize_artifact("worksheet", {"format": "markdown", "markdown": ws})
    assert ws_content["kind"] == "worksheet"
    assert len(ws_content["items"]) >= 4
    assert all(not str(i["question"]).lstrip().startswith("#") for i in ws_content["items"])
    assert "science" in ws_content["items"][0]["question"].lower()
    assert worksheet_items_need_repair(
        [{"question": "# Worksheet: Science\n\n## Warm-Up\n1. A?\n2. B?" * 20}]
    )
    assert not worksheet_items_need_repair(ws_content["items"])
    print("assignments.content selfcheck: ok")


if __name__ == "__main__":
    _selfcheck()

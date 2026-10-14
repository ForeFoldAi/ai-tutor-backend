"""Phase 1: Suneel QA baseline — AI Tutor + AI Voice (voice_mode stream).

Run:
  cd ai-tutor-backend && .venv/bin/python scripts/suneel_qa_baseline.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8000"
VOICE = "http://127.0.0.1:8080"
LOGIN_ID = "Suneel"
LOGIN_PW = "12345678"
OUT_DIR = ROOT / "evaluation" / "reports"
OUT_JSON = OUT_DIR / "suneel_qa_baseline.json"
OUT_MD = OUT_DIR / "suneel_qa_baseline.md"


@dataclass
class QItem:
    id: str
    subject: str
    chapter_id: str
    chapter: str
    class_level: str
    board: str
    query: str
    followup: str
    label: str  # normal | typo | wrong | diagram | interactive
    expect_keywords: list[str]
    forbidden: list[str] = field(default_factory=list)
    want_images: bool = False
    want_math: bool = False
    want_science: bool = False


# 20 primary + follow-ups across Suneel's CLASS_8 catalog
BANK: list[QItem] = [
    # English (4)
    QItem(
        "E1", "English", "26", "Unit 1 - Wit and Wisdom", "CLASS_8", "CBSE",
        "hey what is this unit Wit and Wisdom even about?",
        "ok can you give me one short example from the chapter?",
        "normal", ["wit", "wisdom"],
    ),
    QItem(
        "E2", "English", "27", "Unit 2 - Values and Dispositions", "CLASS_8", "CBSE",
        "what values are they talking about in this chapter?",
        "why do those values matter for students like me?",
        "normal", ["value"],
    ),
    QItem(
        "E3", "English", "28", "Unit 3 - Mystery and Magic", "CLASS_8", "CBSE",
        "is Mystery and Magic a story unit? what happens in it?",
        "quiz me with one easy question on that",
        "normal", ["mystery", "magic"],
    ),
    QItem(
        "E4", "English", "26", "Unit 1 - Wit and Wisdom", "CLASS_8", "CBSE",
        "who invented photosynthesis in Wit and Wisdom?",
        "wait that was wrong right? what should i ask instead?",
        "wrong", ["wit", "wisdom", "chapter", "unit"], ["photosynthesis"],
    ),
    # Mathematics (5)
    QItem(
        "M1", "Mathematics", "38", "Chapter-1 : A SQUARE AND A CUBE", "CLASS_8", "CBSE",
        "wait so what is a perfect square? like in simple words",
        "ok but why do we call it a square number?",
        "normal", ["square"], want_math=True,
    ),
    QItem(
        "M2", "Mathematics", "38", "Chapter-1 : A SQUARE AND A CUBE", "CLASS_8", "CBSE",
        "can you show me how cube numbers work with an example?",
        "show me another example please",
        "interactive", ["cube"], want_math=True,
    ),
    QItem(
        "M3", "Mathematics", "39", "Chapter-2 : POWER PLAY", "CLASS_8", "CBSE",
        "what does power play mean? is it about exponents?",
        "ok explain 2 to the power 5 slowly",
        "normal", ["power", "exponent"], want_math=True,
    ),
    QItem(
        "M4", "Mathematics", "39", "Chapter-2 : POWER PLAY", "CLASS_8", "CBSE",
        "what is the squre of 12? i keep messsing the speling lol",
        "and what about the cube of 5?",
        "typo", ["144", "square", "12"], want_math=True,
    ),
    QItem(
        "M5", "Mathematics", "38", "Chapter-1 : A SQUARE AND A CUBE", "CLASS_8", "CBSE",
        "is the square root of 2 a perfect square in this chapter?",
        "so what IS a perfect square then?",
        "wrong", ["perfect", "square"],
    ),
    # Science (5)
    QItem(
        "S1", "Science", "35", "Chapter 1 - Exploring the Investigative World of Science",
        "CLASS_8", "CBSE",
        "what is this chapter exploring the investigative world of science about?",
        "why do scientists investigate things?",
        "normal", ["science", "investigat"], want_science=True,
    ),
    QItem(
        "S2", "Science", "35", "Chapter 1 - Exploring the Investigative World of Science",
        "CLASS_8", "CBSE",
        "can you show a diagram or figure from this science chapter?",
        "what is that figure trying to show?",
        "diagram", ["figure", "diagram", "image", "shown", "see"], want_images=True, want_science=True,
    ),
    QItem(
        "S3", "Science", "37", "Chapter 3 - Health: The Ultimate Treasure", "CLASS_8", "CBSE",
        "why do they call health the ultimate treasure?",
        "what are some things that keep us healthy according to the chapter?",
        "normal", ["health"], want_science=True,
    ),
    QItem(
        "S4", "Science", "37", "Chapter 3 - Health: The Ultimate Treasure", "CLASS_8", "CBSE",
        "how does disease spread? can you explain like an experiment or steps?",
        "ok but why should i wash my hands then?",
        "interactive", ["disease", "health", "hygiene", "wash", "germ"], want_science=True,
    ),
    QItem(
        "S5", "Science", "35", "Chapter 1 - Exploring the Investigative World of Science",
        "CLASS_8", "CBSE",
        "explain fotosyntesis from this chapter pls",
        "was that even in this chapter or did you guess?",
        "typo", ["chapter", "science", "not", "photosynthesis", "investigat"],
    ),
    # Social (6)
    QItem(
        "So1", "Social", "32", "Chapter 1 - Natural Resources and Their Use", "CLASS_8", "CBSE",
        "what are natural resources? give me easy examples",
        "ok but why should we use them carefully?",
        "normal", ["resource", "natural"],
    ),
    QItem(
        "So2", "Social", "32", "Chapter 1 - Natural Resources and Their Use", "CLASS_8", "CBSE",
        "show me a map or picture about natural resources from the book",
        "what does that picture mean?",
        "diagram", ["resource", "map", "figure", "image", "shown"], want_images=True,
    ),
    QItem(
        "So3", "Social", "33", "Chapter 2 - Reshaping India’s Political Map", "CLASS_8", "CBSE",
        "who were the mughals in this chapter?",
        "can you show the political map figure?",
        "normal", ["mughal"], want_images=True,
    ),
    QItem(
        "So4", "Social", "33", "Chapter 2 - Reshaping India’s Political Map", "CLASS_8", "CBSE",
        "what is a political map anyway?",
        "how is it different from a physical map?",
        "normal", ["political", "map"],
    ),
    QItem(
        "So5", "Social", "34", "Chapter 3 - The Rise of the Marathas", "CLASS_8", "CBSE",
        "who was shivaji and why are the marathas important?",
        "ok tell me one more thing about how the marathas rose",
        "normal", ["maratha", "shivaji"],
    ),
    QItem(
        "So6", "Social", "34", "Chapter 3 - The Rise of the Marathas", "CLASS_8", "CBSE",
        "did the marathas invent the steam engine in this chapter?",
        "lol ok so what DID they actually do?",
        "wrong", ["maratha"], ["steam engine", "steam"],
    ),
]


def http_json(method: str, url: str, body: dict | None = None, token: str | None = None, timeout: int = 180):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} -> {e.code}: {err[:300]}") from e


def login() -> str:
    for ident in (LOGIN_ID, LOGIN_ID.lower(), "kammara.suneel474@gmail.com"):
        try:
            res = http_json("POST", f"{API}/auth/login", {"email": ident, "password": LOGIN_PW})
            return res["access_token"]
        except Exception as e:
            last = e
    raise RuntimeError(f"login failed: {last}")


def clip(s: str | None, n: int = 400) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def img_brief(imgs: list | None) -> list[dict]:
    out = []
    for im in imgs or []:
        out.append({
            "figure": im.get("figure_number"),
            "page": im.get("page"),
            "caption": clip(im.get("caption") or "", 80),
            "file": im.get("file_name"),
        })
    return out


def score_turn(
    item: QItem,
    *,
    answer: str,
    images: list | None,
    math_lesson,
    science_experiment,
    channel: str,
) -> dict:
    a = (answer or "").lower()
    bugs: list[str] = []

    # Answer quality (heuristic)
    ans = 5.0
    if len(a) < 40:
        ans = 2.0
        bugs.append("very short answer")
    elif len(a) > 80:
        ans = 7.0
    hits = sum(1 for k in item.expect_keywords if k.lower() in a)
    if hits:
        ans = min(10.0, ans + 1.5 * hits)
    else:
        ans = max(1.0, ans - 3.0)
        bugs.append("missing expected keywords")
    for f in item.forbidden:
        if f.lower() in a:
            ans = max(1.0, ans - 2.5)
            bugs.append(f"forbidden content mentioned: {f}")
    if item.label in ("wrong", "typo") and ("not" in a or "doesn't" in a or "does not" in a or "isn't" in a or "chapter" in a):
        ans = min(10.0, ans + 1.0)

    # Expected vs actual (keyword coverage)
    denom = max(1, len(item.expect_keywords))
    expect = round(10.0 * hits / denom, 1)

    # Images
    nimg = len(images or [])
    if item.want_images:
        if nimg >= 1:
            images_score = 9.0
        else:
            images_score = 2.0
            bugs.append("wanted images but got none")
    else:
        images_score = 8.0 if nimg >= 0 else 8.0
        if nimg:
            images_score = 9.0

    # Interactive
    if channel == "voice":
        if item.want_math or item.want_science:
            interactive = None  # N/A — Nest drops panels; score via answer teaching quality
            if len(a) > 100:
                interactive_note = "N/A (voice); transcript length ok"
            else:
                interactive_note = "N/A (voice); thin transcript"
                bugs.append("voice transcript thin for interactive topic")
        else:
            interactive = None
            interactive_note = "N/A"
    else:
        interactive_note = ""
        if item.want_math:
            if math_lesson:
                interactive = 9.0
            else:
                interactive = 3.0
                bugs.append("expected math_lesson missing")
        elif item.want_science:
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


def chat_tutor(token: str, item: QItem, query: str, history: list[dict] | None) -> dict:
    body = {
        "query": query,
        "board": item.board,
        "class_level": item.class_level,
        "subject_name": item.subject,
        "chapter_ids": [item.chapter_id],
        "chapter": item.chapter,
        "chapter_names": [item.chapter],
        "conversation_history": history or [],
    }
    t0 = time.time()
    res = http_json("POST", f"{API}/auth/chat", body, token=token, timeout=240)
    return {
        "answer": res.get("answer") or "",
        "related_images": res.get("related_images") or [],
        "math_lesson": res.get("math_lesson"),
        "science_experiment": res.get("science_experiment"),
        "ms": int((time.time() - t0) * 1000),
        "error": None,
    }


def chat_voice_stream(token: str, item: QItem, query: str, history: list[dict] | None) -> dict:
    """Same path Nest rag.client uses: /auth/chat/stream with voice_mode=true."""
    body = {
        "query": query,
        "board": item.board,
        "class_level": item.class_level,
        "subject_name": item.subject,
        "chapter_ids": [item.chapter_id],
        "chapter": item.chapter,
        "chapter_names": [item.chapter],
        "conversation_history": history or [],
        "voice_mode": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}/auth/chat/stream",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    t0 = time.time()
    tokens: list[str] = []
    clean = ""
    images: list = []
    math_lesson = None
    science_experiment = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            for line in r:
                line = line.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = ev.get("type")
                if typ == "token":
                    tokens.append(ev.get("content") or "")
                elif typ == "clean_answer":
                    clean = ev.get("content") or clean
                elif typ == "related_images":
                    images = ev.get("images") or images
                elif typ == "math_lesson":
                    math_lesson = ev.get("lesson")
                    if ev.get("clean_answer"):
                        clean = ev["clean_answer"]
                elif typ == "science_experiment":
                    science_experiment = ev.get("experiment")
                    if ev.get("clean_answer"):
                        clean = ev["clean_answer"]
    except Exception as e:
        err = str(e)[:300]
    answer = clean or "".join(tokens)
    return {
        "answer": answer,
        "related_images": images,
        "math_lesson": math_lesson,
        "science_experiment": science_experiment,
        "ms": int((time.time() - t0) * 1000),
        "error": err,
        "path": "auth/chat/stream voice_mode=true",
    }


def probe_nest(token: str, item: QItem) -> dict:
    """Verify Nest session accepts JWT + chapter scope (answer via WS only)."""
    try:
        health = http_json("GET", f"{VOICE}/rtc/health")
    except Exception as e:
        return {"ok": False, "error": f"health: {e}"}
    try:
        sess = http_json(
            "POST",
            f"{VOICE}/rtc/voice/session",
            {
                "board": item.board,
                "classLevel": item.class_level,
                "subject": item.subject,
                "chapterIds": [item.chapter_id],
                "chapterNames": [item.chapter],
                "chapter": item.chapter,
            },
            token=token,
            timeout=30,
        )
        sid = sess.get("id")
        # REST query returns ok only; do not rely on it for answer text
        q = http_json(
            "POST",
            f"{VOICE}/rtc/tutor/query",
            {"sessionId": sid, "query": "hi just checking voice works"},
            token=token,
            timeout=60,
        )
        try:
            http_json("POST", f"{VOICE}/rtc/voice/session/{sid}/end", {}, token=token, timeout=15)
        except Exception:
            pass
        return {"ok": True, "health": health, "session_id": sid, "query_ok": q.get("ok"), "note": "REST returns ok only; answers scored via voice_mode stream"}
    except Exception as e:
        return {"ok": False, "health": health, "error": str(e)[:300]}


def run_channel(token: str, channel: str, runner) -> list[dict]:
    rows = []
    for i, item in enumerate(BANK, 1):
        print(f"[{channel}] {i}/{len(BANK)} {item.id} primary…", flush=True)
        try:
            primary = runner(token, item, item.query, None)
        except Exception as e:
            primary = {"answer": "", "related_images": [], "math_lesson": None, "science_experiment": None, "ms": 0, "error": str(e)[:300]}
        p_score = score_turn(
            item,
            answer=primary.get("answer") or "",
            images=primary.get("related_images"),
            math_lesson=primary.get("math_lesson"),
            science_experiment=primary.get("science_experiment"),
            channel=channel,
        )
        history = [
            {"role": "user", "content": item.query},
            {"role": "assistant", "content": primary.get("answer") or ""},
        ]
        print(f"[{channel}] {item.id} follow-up…", flush=True)
        try:
            follow = runner(token, item, item.followup, history)
        except Exception as e:
            follow = {"answer": "", "related_images": [], "math_lesson": None, "science_experiment": None, "ms": 0, "error": str(e)[:300]}
        f_score = score_turn(
            item,
            answer=follow.get("answer") or "",
            images=follow.get("related_images"),
            math_lesson=follow.get("math_lesson"),
            science_experiment=follow.get("science_experiment"),
            channel=channel,
        )
        rows.append({
            "id": item.id,
            "subject": item.subject,
            "chapter": item.chapter,
            "chapter_id": item.chapter_id,
            "label": item.label,
            "query": item.query,
            "followup": item.followup,
            "expect_keywords": item.expect_keywords,
            "primary": {
                "answer": primary.get("answer") or "",
                "answer_clip": clip(primary.get("answer"), 500),
                "images": img_brief(primary.get("related_images")),
                "n_images": len(primary.get("related_images") or []),
                "has_math_lesson": bool(primary.get("math_lesson")),
                "has_science_experiment": bool(primary.get("science_experiment")),
                "ms": primary.get("ms"),
                "error": primary.get("error"),
                "path": primary.get("path"),
                "scores": p_score,
            },
            "follow": {
                "answer": follow.get("answer") or "",
                "answer_clip": clip(follow.get("answer"), 500),
                "images": img_brief(follow.get("related_images")),
                "n_images": len(follow.get("related_images") or []),
                "has_math_lesson": bool(follow.get("math_lesson")),
                "has_science_experiment": bool(follow.get("science_experiment")),
                "ms": follow.get("ms"),
                "error": follow.get("error"),
                "path": follow.get("path"),
                "scores": f_score,
            },
            "item_overall": round((p_score["overall"] + f_score["overall"]) / 2, 1),
        })
        print(f"[{channel}] {item.id} overall={rows[-1]['item_overall']}", flush=True)
    return rows


def write_md(payload: dict) -> None:
    tutor = payload["tutor"]
    voice = payload["voice"]
    nest = payload.get("nest_probe") or {}
    cat = payload.get("catalog_summary") or {}

    def avg(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return round(sum(r["item_overall"] for r in rows) / len(rows), 2)

    lines: list[str] = []
    lines.append("# Suneel QA Baseline — AI Tutor + AI Voice")
    lines.append("")
    lines.append(f"- **Date:** {payload.get('generated_at')}")
    lines.append(f"- **Login:** `{LOGIN_ID}` (password not recorded)")
    lines.append(f"- **Student:** {cat.get('full_name')} · role={cat.get('role')} · board={cat.get('board')} · class=CLASS_8 · school_id={cat.get('school_id')}")
    lines.append(f"- **Services:** FastAPI `{API}` · Nest voice `{VOICE}` (health ok={nest.get('ok')})")
    lines.append(f"- **Voice scoring path:** `{payload.get('voice_path')}`")
    if nest.get("note"):
        lines.append(f"- **Nest note:** {nest.get('note')}")
    if nest.get("error"):
        lines.append(f"- **Nest probe error:** {nest.get('error')}")
    lines.append("")
    lines.append("## Subject / chapter inventory")
    lines.append("")
    for sub, chs in (cat.get("subjects") or {}).items():
        lines.append(f"- **{sub}:** " + "; ".join(f"`{c['id']}` {c['chapter']}" for c in chs))
    lines.append("")
    lines.append("## Question bank (20 + follow-ups)")
    lines.append("")
    lines.append("| ID | Label | Subject | Chapter | Primary | Follow-up |")
    lines.append("|----|-------|---------|---------|---------|-----------|")
    for item in BANK:
        lines.append(
            f"| {item.id} | {item.label} | {item.subject} | {item.chapter} | {item.query} | {item.followup} |"
        )
    lines.append("")
    lines.append(f"## Overall ranks")
    lines.append("")
    lines.append(f"- **AI Tutor average:** **{avg(tutor)} / 10**")
    lines.append(f"- **AI Voice average:** **{avg(voice)} / 10**")
    lines.append("")
    lines.append("## Per-question dual results")
    lines.append("")

    by_v = {r["id"]: r for r in voice}
    for t in tutor:
        v = by_v.get(t["id"], {})
        lines.append(f"### {t['id']} — {t['subject']} / {t['label']}")
        lines.append("")
        lines.append(f"- **Chapter:** {t['chapter']} (`{t['chapter_id']}`)")
        lines.append(f"- **Q:** {t['query']}")
        lines.append(f"- **Follow-up:** {t['followup']}")
        lines.append(f"- **Expected keywords:** {', '.join(t['expect_keywords'])}")
        lines.append("")
        lines.append("| Channel | Turn | Overall | Ans | Expect | Images | Interactive | n_images | math | science | Bugs |")
        lines.append("|---------|------|---------|-----|--------|--------|-------------|----------|------|---------|------|")
        for ch_name, row in (("Tutor", t), ("Voice", v)):
            if not row:
                continue
            for turn_key, label in (("primary", "primary"), ("follow", "follow-up")):
                turn = row[turn_key]
                sc = turn["scores"]
                inter = sc["interactive_score"]
                inter_s = "N/A" if inter is None else str(inter)
                if sc.get("interactive_note"):
                    inter_s = f"{inter_s} ({sc['interactive_note']})"
                bugs = "; ".join(sc.get("bugs") or []) or (turn.get("error") or "—")
                lines.append(
                    f"| {ch_name} | {label} | {sc['overall']} | {sc['answer_score']} | {sc['expect_score']} | "
                    f"{sc['images_score']} | {inter_s} | {turn['n_images']} | {turn['has_math_lesson']} | "
                    f"{turn['has_science_experiment']} | {bugs} |"
                )
        lines.append("")
        lines.append(f"- **Tutor primary (clip):** {t['primary']['answer_clip']}")
        lines.append(f"- **Tutor follow (clip):** {t['follow']['answer_clip']}")
        if v:
            lines.append(f"- **Voice primary (clip):** {v['primary']['answer_clip']}")
            lines.append(f"- **Voice follow (clip):** {v['follow']['answer_clip']}")
        if t["primary"]["images"]:
            lines.append(f"- **Tutor images:** `{json.dumps(t['primary']['images'])}`")
        if v and v["primary"]["images"]:
            lines.append(f"- **Voice images:** `{json.dumps(v['primary']['images'])}`")
        lines.append(f"- **Item overall Tutor / Voice:** {t['item_overall']} / {v.get('item_overall', '—')}")
        lines.append("")

    # Ranked worst→best
    lines.append("## Ranked summary (worst → best)")
    lines.append("")
    lines.append("### AI Tutor")
    for r in sorted(tutor, key=lambda x: x["item_overall"]):
        lines.append(f"- `{r['id']}` {r['item_overall']}/10 — {r['subject']} — {r['label']}")
    lines.append("")
    lines.append("### AI Voice")
    for r in sorted(voice, key=lambda x: x["item_overall"]):
        lines.append(f"- `{r['id']}` {r['item_overall']}/10 — {r['subject']} — {r['label']}")
    lines.append("")

    # Channel gaps
    lines.append("### Largest Tutor↔Voice gaps")
    gaps = []
    for t in tutor:
        v = by_v.get(t["id"])
        if not v:
            continue
        gaps.append((abs(t["item_overall"] - v["item_overall"]), t["id"], t["item_overall"], v["item_overall"]))
    for g, i, tv, vv in sorted(gaps, reverse=True)[:8]:
        lines.append(f"- `{i}` gap={g:.1f} (Tutor {tv} vs Voice {vv})")
    lines.append("")

    # Bugs & suggested fixes
    all_bugs: dict[str, int] = {}
    for rows, ch in ((tutor, "tutor"), (voice, "voice")):
        for r in rows:
            for turn in (r["primary"], r["follow"]):
                for b in turn["scores"].get("bugs") or []:
                    all_bugs[f"[{ch}] {b}"] = all_bugs.get(f"[{ch}] {b}", 0) + 1
                if turn.get("error"):
                    k = f"[{ch}] error: {turn['error'][:80]}"
                    all_bugs[k] = all_bugs.get(k, 0) + 1

    lines.append("## Bugs & suggested fixes (no code changes in Phase 1)")
    lines.append("")
    lines.append("| Frequency | Bug signal | Suggested fix locus |")
    lines.append("|-----------|------------|---------------------|")
    for bug, n in sorted(all_bugs.items(), key=lambda x: -x[1]):
        locus = "prompts / retrieval"
        if "images" in bug:
            locus = "`chat_service/images.py`, `textbook_image_retrieval.py`"
        elif "math_lesson" in bug:
            locus = "`math_lesson/service.py`, math section of `prompts.py`"
        elif "science_experiment" in bug:
            locus = "`science_experiment/service.py`, science prompts"
        elif "voice" in bug.lower() or "transcript" in bug:
            locus = "`Voice/backend/.../rag.client.ts` (forward panels) + voice prompts"
        elif "forbidden" in bug or "wrong" in bug:
            locus = "`chapter_scope.py` / RAG filter + refusal guidance in `prompts.py`"
        elif "keyword" in bug:
            locus = "retrieval / `qa.py` grounding"
        lines.append(f"| {n} | {bug} | {locus} |")

    lines.append("")
    lines.append("### Priority fix themes")
    lines.append("")
    lines.append("1. **Grounding / wrong-premise handling** — adversarial & typo items should correct the student without inventing off-chapter facts (`prompts.py`, `chapter_scope.py`).")
    lines.append("2. **Images on diagram asks** — Science/Social diagram questions need reliable `related_images` (`images.py`, CLIP/BGE rank).")
    lines.append("3. **Math/Science interactive JSON** — Tutor path should emit `math_lesson` / `science_experiment` when asked; Voice Nest currently drops these — consider forwarding in `rag.client.ts` or teaching fully in spoken prose.")
    lines.append("4. **Follow-up continuity** — follow-ups must continue teaching with history (`conversation_context.py`, Nest recentMessages).")
    lines.append("5. **Tutor↔Voice parity** — same question should not drift; share postprocess/`clean_answer` behavior under `voice_mode`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Phase 1 baseline only. Return this MD for a Phase 2 code-fix plan. Target after fixes: ≥ 9/10 on both channels.*")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("logging in…", flush=True)
    token = login()
    me = http_json("GET", f"{API}/auth/me", token=token)
    subjects = http_json("GET", f"{API}/auth/catalog/my-subjects", token=token)
    cat_subjects = {}
    for s in subjects:
        cat_subjects[s["subject_name"]] = [
            {"id": c["id"], "chapter": c["chapter"]} for c in (s.get("chapters") or [])
        ]
    catalog_summary = {
        "full_name": me.get("full_name"),
        "role": me.get("role"),
        "board": me.get("teaching_board") or "CBSE",
        "school_id": me.get("school_id"),
        "subjects": cat_subjects,
    }
    print("subjects:", list(cat_subjects), flush=True)

    nest = probe_nest(token, BANK[0])
    print("nest_probe:", nest, flush=True)

    print("=== AI TUTOR ===", flush=True)
    tutor_rows = run_channel(token, "tutor", chat_tutor)
    print("=== AI VOICE (voice_mode stream) ===", flush=True)
    voice_rows = run_channel(token, "voice", chat_voice_stream)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "catalog_summary": catalog_summary,
        "nest_probe": nest,
        "voice_path": "POST /auth/chat/stream voice_mode=true (Nest rag.client equivalent; REST /rtc/tutor/query is fire-and-forget)",
        "tutor": tutor_rows,
        "voice": voice_rows,
        "tutor_avg": round(sum(r["item_overall"] for r in tutor_rows) / max(1, len(tutor_rows)), 2),
        "voice_avg": round(sum(r["item_overall"] for r in voice_rows) / max(1, len(voice_rows)), 2),
    }
    # Trim full answers in JSON companion to keep file smaller but keep clips + scores
    slim = json.loads(json.dumps(payload))
    for ch in ("tutor", "voice"):
        for r in slim[ch]:
            r["primary"].pop("answer", None)
            r["follow"].pop("answer", None)
    OUT_JSON.write_text(json.dumps(slim, indent=2))
    write_md(payload)
    print("Wrote", OUT_MD)
    print("Wrote", OUT_JSON)
    print("TUTOR_AVG", payload["tutor_avg"], "VOICE_AVG", payload["voice_avg"])


if __name__ == "__main__":
    main()

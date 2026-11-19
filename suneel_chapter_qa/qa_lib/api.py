"""HTTP client for the benchmark: login, Tutor / Voice / Ask Learn Mastro chat, Nest probe.

Env overrides: QA_API (default http://127.0.0.1:8000), QA_VOICE (http://127.0.0.1:8080),
QA_LOGIN, QA_PASSWORD.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API = os.environ.get("QA_API", "http://127.0.0.1:8000").rstrip("/")
VOICE = os.environ.get("QA_VOICE", "http://127.0.0.1:8080").rstrip("/")
LOGIN_IDS = [x for x in (os.environ.get("QA_LOGIN"), "Suneel", "suneel", "kammara.suneel474@gmail.com") if x]
LOGIN_PW = os.environ.get("QA_PASSWORD", "12345678")
CHAT_TIMEOUT = int(os.environ.get("QA_CHAT_TIMEOUT", "240"))


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


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
        raise HttpError(e.code, f"{method} {url} -> {e.code}: {err[:300]}") from e


def require_api() -> None:
    try:
        urllib.request.urlopen(f"{API}/health", timeout=10).read()
    except urllib.error.HTTPError:
        return  # server answered (even 404) — it is up
    except Exception as e:
        raise SystemExit(f"API not reachable at {API}: {e}") from e


class AuthSession:
    """Holds a JWT and re-logs in when it expires (long runs outlive the token)."""

    def __init__(self) -> None:
        self.token = ""
        self.login_as = ""
        self.login()

    def login(self) -> str:
        last: Exception | None = None
        for ident in LOGIN_IDS:
            try:
                res = http_json("POST", f"{API}/auth/login", {"email": ident, "password": LOGIN_PW})
                self.token = res["access_token"]
                self.login_as = ident
                return self.token
            except Exception as e:
                last = e
        raise RuntimeError(f"login failed: {last}")

    def json(self, method: str, url: str, body: dict | None = None, timeout: int = 180):
        try:
            return http_json(method, url, body, token=self.token, timeout=timeout)
        except HttpError as e:
            if e.status != 401:
                raise
            self.login()
            return http_json(method, url, body, token=self.token, timeout=timeout)

    def open_stream(self, url: str, body: dict, timeout: int = CHAT_TIMEOUT):
        def _open():
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            )
            return urllib.request.urlopen(req, timeout=timeout)

        try:
            return _open()
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            self.login()
            return _open()


def clip(s: str | None, n: int = 400) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def img_brief(imgs: list | None) -> list[dict]:
    out = []
    for im in imgs or []:
        out.append(
            {
                "figure": im.get("figure_number"),
                "page": im.get("page"),
                "caption": clip(im.get("caption") or "", 80),
                "file": im.get("file_name"),
            }
        )
    return out


def _chapter_body(query, history, board, class_level, subject, chapter_id, chapter) -> dict:
    return {
        "query": query,
        "board": board,
        "class_level": class_level,
        "subject_name": subject,
        "chapter_ids": [str(chapter_id)],
        "chapter": chapter,
        "chapter_names": [chapter],
        "conversation_history": history or [],
    }


def _read_ndjson_stream(session: AuthSession, url: str, body: dict, path: str) -> dict:
    """Consume an NDJSON chat stream; record total time and time to first answer token."""
    t0 = time.time()
    ttft_ms: int | None = None
    tokens: list[str] = []
    clean = ""
    images: list = []
    math_lesson = None
    science_experiment = None
    err = None
    meta: dict = {}
    try:
        with session.open_stream(url, body) as r:
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
                    if ttft_ms is None and (ev.get("content") or "").strip():
                        ttft_ms = int((time.time() - t0) * 1000)
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
                elif typ == "error":
                    err = (ev.get("content") or "stream error")[:300]
                elif typ == "done":
                    meta = {k: v for k, v in ev.items() if k != "type"}
    except Exception as e:
        err = str(e)[:300]
    return {
        "meta": meta,
        "answer": clean or "".join(tokens),
        "related_images": images,
        "math_lesson": math_lesson,
        "science_experiment": science_experiment,
        "ms": int((time.time() - t0) * 1000),
        "ttft_ms": ttft_ms,
        "error": err,
        "path": path,
    }


def chat_tutor(
    session: AuthSession,
    *,
    query: str,
    history: list[dict] | None,
    board: str,
    class_level: str,
    subject: str,
    chapter_id,
    chapter: str,
    extra: dict | None = None,
) -> dict:
    """AI Tutor text chat — same stream endpoint the /ai-tutor page uses.
    *extra* carries the per-chat state the page sends back (quiz_state, prefers_short, explained_points)."""
    body = _chapter_body(query, history, board, class_level, subject, chapter_id, chapter)
    body.update(extra or {})
    return _read_ndjson_stream(session, f"{API}/auth/chat/stream", body, "POST /auth/chat/stream")


def chat_voice(
    session: AuthSession,
    *,
    query: str,
    history: list[dict] | None,
    board: str,
    class_level: str,
    subject: str,
    chapter_id,
    chapter: str,
    extra: dict | None = None,
) -> dict:
    """Same path Nest rag.client uses: /auth/chat/stream with voice_mode=true."""
    body = _chapter_body(query, history, board, class_level, subject, chapter_id, chapter)
    body.update(extra or {})
    body["voice_mode"] = True
    return _read_ndjson_stream(
        session, f"{API}/auth/chat/stream", body, "POST /auth/chat/stream voice_mode=true"
    )


def chat_assistant(
    session: AuthSession,
    *,
    query: str,
    history: list[dict] | None,
    board: str,
    class_level: str,
    subject: str,
    chapter_id,
    chapter: str,
    agent_mode: str = "ask",
) -> dict:
    """Ask Learn Mastro (student assistant). It has no chapter picker, so a leading
    user turn names the subject and chapter the way a student would."""
    scope_turn = {"role": "user", "content": f"I am studying {subject}, {chapter}."}
    body = {
        "query": query,
        "conversation_history": [scope_turn, *(history or [])],
        "agent_mode": agent_mode,
    }
    return _read_ndjson_stream(
        session,
        f"{API}/auth/student/assistant/chat/stream",
        body,
        f"POST /auth/student/assistant/chat/stream mode={agent_mode}",
    )


def probe_nest(session: AuthSession, bank: dict) -> dict:
    """Verify Nest session accepts JWT + chapter scope (answer via WS only)."""
    try:
        health = http_json("GET", f"{VOICE}/rtc/health")
    except Exception as e:
        return {"ok": False, "error": f"health: {e}"}
    try:
        sess = session.json(
            "POST",
            f"{VOICE}/rtc/voice/session",
            {
                "board": bank["board"],
                "classLevel": bank["class_level"],
                "subject": bank["subject"],
                "chapterIds": [str(bank["chapter_id"])],
                "chapterNames": [bank["chapter"]],
                "chapter": bank["chapter"],
            },
            timeout=30,
        )
        sid = sess.get("id")
        q = session.json(
            "POST",
            f"{VOICE}/rtc/tutor/query",
            {"sessionId": sid, "query": "hi just checking voice works"},
            timeout=60,
        )
        try:
            session.json("POST", f"{VOICE}/rtc/voice/session/{sid}/end", {}, timeout=15)
        except Exception:
            pass
        return {
            "ok": True,
            "health": health,
            "session_id": sid,
            "query_ok": q.get("ok"),
            "note": "REST ok only; scored via voice_mode stream",
        }
    except Exception as e:
        return {"ok": False, "health": health, "error": str(e)[:300]}

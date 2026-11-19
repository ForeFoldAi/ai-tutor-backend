"""Upload single-chapter PDFs (Direct / Single Chapter flow) and time processing end to end.

Usage:
  ADMIN_USER=master ADMIN_PASSWORD=... .venv/bin/python scripts/upload_chapters_prod.py

Env (optional): API_URL (default prod), PDF_DIR, POLL_SECONDS, TIMEOUT_MINUTES.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

API = os.environ.get("API_URL", "https://api.tutor.forefoldai.com").rstrip("/")
PDF_DIR = Path(os.environ.get("PDF_DIR", "/Users/manithmettu/Downloads/jesc1dd"))
POLL = int(os.environ.get("POLL_SECONDS", "10"))
TIMEOUT = int(os.environ.get("TIMEOUT_MINUTES", "60")) * 60

BOARD, CLASS, SUBJECT = "CBSE", "CLASS_10", "Science"
CHAPTERS = [
    ("jesc101.pdf", "Chapter 1 Chemical Reactions and Equations"),
    ("jesc102.pdf", "Chapter 2 Acids, Bases and Salts"),
    ("jesc103.pdf", "Chapter 3 Metals and Non-metals"),
    ("jesc104.pdf", "Chapter 4 Carbon and its Compounds"),
    ("jesc105.pdf", "Chapter 5 Life Processes"),
]
DONE = {"EMBEDDED", "FAILED"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def login(c: httpx.Client) -> None:
    user, pw = os.environ.get("ADMIN_USER"), os.environ.get("ADMIN_PASSWORD")
    if not user or not pw:
        sys.exit("Set ADMIN_USER and ADMIN_PASSWORD (master admin).")
    r = c.post("/auth/login", json={"email": user, "password": pw})
    r.raise_for_status()
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"


def worker_snapshot(c: httpx.Client) -> dict:
    r = c.get("/auth/admin/catalog/worker-status")
    return r.json() if r.is_success else {"error": f"{r.status_code} {r.text[:200]}"}


def print_workers(ws: dict) -> None:
    if "error" in ws:
        log(f"worker-status error: {ws['error']}")
        return
    log(
        f"status={ws['status']} redis={ws['redis_connected']} queue={ws['queue_depths']} "
        f"processing={ws['processing_count']} queued={ws['queued_count']} failed={ws['failed_count']}"
    )
    for w in ws["workers"]:
        log(f"  worker {w['name']} {w['status']} concurrency={w['concurrency']} active={w['active_tasks_count']} queues={w['active_queues']}")
    for j in ws["active_jobs"]:
        log(f"  active {j['task_name']} {j['target_label']} on {j['worker']}")


def main() -> None:
    missing = [f for f, _ in CHAPTERS if not (PDF_DIR / f).is_file()]
    if missing:
        sys.exit(f"Missing PDFs in {PDF_DIR}: {missing}")

    with httpx.Client(base_url=API, timeout=httpx.Timeout(30, write=600, read=600)) as c:
        login(c)
        log(f"Logged in to {API}")
        print_workers(worker_snapshot(c))

        t0 = time.monotonic()
        rows: dict[int, dict] = {}
        for fname, label in CHAPTERS:
            path = PDF_DIR / fname
            started = time.monotonic()
            with path.open("rb") as fh:
                r = c.post(
                    "/auth/admin/catalog/textbook-uploads/upload",
                    files={"files": (fname, fh, "application/pdf")},
                    # ponytail: no textbook_id = "default" publisher pool, same as the UI.
                    data={"board": BOARD, "class_name": CLASS, "subject": SUBJECT,
                          "content_type": "CHAPTER", "content_labels": label},
                )
            if not r.is_success:
                log(f"UPLOAD FAILED {fname}: {r.status_code} {r.text[:300]}")
                continue
            row = r.json()[0]
            up_s = time.monotonic() - started
            mb = path.stat().st_size / 1e6
            log(f"Uploaded {fname} ({mb:.1f} MB) in {up_s:.1f}s -> id={row['id']} label={row['chapter']!r}")
            if row["chapter"] != label:
                # Backend splits content_labels on commas, so "Acids, Bases and Salts" gets cut; fix it.
                log(f"  label truncated to {row['chapter']!r}; correcting via PUT")
                c.put(f"/auth/admin/catalog/textbook-uploads/{row['id']}", json={"chapter": label, "content_label": label})
            rows[row["id"]] = {"file": fname, "mb": mb, "upload_s": up_s, "submitted": time.monotonic(),
                               "started": None, "finished": None, "status": None}

        log(f"All uploads sent in {time.monotonic() - t0:.1f}s; polling every {POLL}s")
        last_ws = 0.0
        while any(v["finished"] is None for v in rows.values()):
            if time.monotonic() - t0 > TIMEOUT:
                log("TIMEOUT waiting for processing")
                break
            try:
                r = c.get("/auth/admin/catalog/textbook-uploads")
            except httpx.HTTPError as exc:
                log(f"poll error: {exc}")
                time.sleep(POLL)
                continue
            if r.status_code == 401:
                login(c)  # access token expired (15 min default)
                continue
            if not r.is_success:
                log(f"poll error: {r.status_code} {r.text[:200]}")
                time.sleep(POLL)
                continue
            listing = {u["id"]: u for u in r.json()}
            for uid, v in rows.items():
                u = listing.get(uid)
                if not u or v["finished"]:
                    continue
                status = (u["ocr_status"], u["chunk_status"], u["embedding_status"])
                if status != v["status"]:
                    log(f"  #{uid} {v['file']}: ocr={status[0]} chunk={status[1]} embed={status[2]} chunks={u['chunk_count']}")
                    v["status"] = status
                if v["started"] is None and "PROCESSING" in status:
                    v["started"] = time.monotonic()
                if u["embedding_status"] in DONE and u["chunk_status"] in DONE:
                    v["finished"] = time.monotonic()
                    v["chunks"] = u["chunk_count"]
                    v["result"] = u["embedding_status"]
            if time.monotonic() - last_ws > 60:
                print_workers(worker_snapshot(c))
                last_ws = time.monotonic()
            time.sleep(POLL)

        print("\n=== Summary ===")
        print(f"{'id':>6} {'file':<12} {'MB':>5} {'upload':>7} {'wait':>7} {'process':>8} {'total':>7} {'chunks':>6} result")
        for uid, v in rows.items():
            wait = (v["started"] or v["finished"] or time.monotonic()) - v["submitted"]
            proc = (v["finished"] - v["started"]) if v["finished"] and v["started"] else None
            total = (v["finished"] - v["submitted"] + v["upload_s"]) if v["finished"] else None
            fmt = lambda s: f"{s:.0f}s" if s is not None else "-"
            print(f"{uid:>6} {v['file']:<12} {v['mb']:>5.1f} {fmt(v['upload_s']):>7} {fmt(wait):>7} {fmt(proc):>8} "
                  f"{fmt(total):>7} {v.get('chunks', '-'):>6} {v.get('result', 'NOT DONE')}")
        print(f"Wall clock: {time.monotonic() - t0:.0f}s")
        ws = worker_snapshot(c)
        print_workers(ws)
        Path("upload_chapters_prod_result.json").write_text(json.dumps({"rows": rows, "worker_status": ws}, default=str, indent=2))


if __name__ == "__main__":
    main()

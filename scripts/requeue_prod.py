"""Requeue Queued + Stuck textbook uploads (same rules as the admin UI).

Usage (dry run lists only; add --apply to requeue):
  ADMIN_USER=... ADMIN_PASSWORD=... .venv/bin/python scripts/requeue_prod.py [--apply]
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import httpx

API = os.environ.get("API_URL", "https://api.tutor.forefoldai.com").rstrip("/")
STUCK_AFTER = timedelta(minutes=40)  # keep in sync with STUCK_AFTER_MS in upload-progress.ts


def ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def stage(u: dict, now: datetime) -> str:
    if "FAILED" in (u["chunk_status"], u["embedding_status"]):
        return "failed"
    started = ts(u.get("processing_started_at"))
    if u["embedding_status"] == "EMBEDDED":
        if not started or u.get("processing_finished_at"):
            return "completed"
        return "stuck" if now - started > STUCK_AFTER else "images"
    if "PROCESSING" in (u["chunk_status"], u["embedding_status"]):
        since = started or ts(u["upload_date"])
        return "stuck" if now - since > STUCK_AFTER else "running"
    return "queued"


def main() -> None:
    apply = "--apply" in sys.argv
    user, pw = os.environ.get("ADMIN_USER"), os.environ.get("ADMIN_PASSWORD")
    if not user or not pw:
        sys.exit("Set ADMIN_USER and ADMIN_PASSWORD (master admin).")
    with httpx.Client(base_url=API, timeout=60) as c:
        r = c.post("/auth/login", json={"email": user, "password": pw})
        r.raise_for_status()
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"

        rows = c.get("/auth/admin/catalog/textbook-uploads").raise_for_status().json()
        now = datetime.now(UTC)
        print(f"{len(rows)} uploads; timing columns deployed: {bool(rows) and 'processing_started_at' in rows[0]}")
        targets = []
        for u in rows:
            s = stage(u, now)
            if s in ("queued", "stuck"):
                targets.append(u)
                print(f"  {s:6} id={u['id']} {u['board']} {u['class_level']} {u['subject_name']!r} "
                      f"{u.get('content_label') or u['file_name']!r} uploaded={u['upload_date']} "
                      f"ocr/chunk/emb={u['ocr_status']}/{u['chunk_status']}/{u['embedding_status']}")
        print(f"{len(targets)} to requeue")

        ws = c.get("/auth/admin/catalog/worker-status")
        if ws.is_success:
            w = ws.json()
            print(f"worker: status={w.get('status')} redis={w.get('redis_connected')} "
                  f"broker_queue={w.get('queue_depths')} workers={[(x['name'], x['status']) for x in w.get('workers') or []]} "
                  f"active={[j.get('target_label') for j in w.get('active_jobs') or []]}")

        if not apply:
            print("dry run; pass --apply to requeue")
            return
        for u in sorted(targets, key=lambda u: u["upload_date"]):  # worker runs FIFO, one at a time
            r = c.post(f"/auth/admin/catalog/textbook-uploads/{u['id']}/process")
            print(f"  requeue id={u['id']}: {r.status_code} {r.text[:150] if not r.is_success else ''}")


if __name__ == "__main__":
    main()

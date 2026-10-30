"""Fire N concurrent tutor chats and report time-to-first-token plus /health latency.

Usage:
  python scripts/chat_load_check.py --base https://api.example.com \
      --email ssc8 --password 12345678 --n 10 \
      --board "State Board" --class-level CLASS_8 --subject Mathematics \
      --chapter-id <upload_id> --chapter-name "Chapter 1"

A healthy server keeps /health under ~100 ms while chats stream, and first-token
time grows gently with N instead of stacking up one chat after another.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def one_chat(client: httpx.AsyncClient, body: dict, headers: dict) -> dict:
    t0 = time.perf_counter()
    out = {"status": None, "first_token": None, "total": None, "error": None}
    try:
        async with client.stream("POST", "/auth/chat/stream", json=body, headers=headers) as r:
            if r.status_code != 200:
                out["error"] = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return out
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                evt = json.loads(line)
                elapsed = time.perf_counter() - t0
                if evt.get("type") == "status" and out["status"] is None:
                    out["status"] = elapsed
                elif evt.get("type") == "token" and out["first_token"] is None:
                    out["first_token"] = elapsed
    except Exception as exc:
        out["error"] = repr(exc)
    out["total"] = time.perf_counter() - t0
    return out


async def poll_health(client: httpx.AsyncClient, stop: asyncio.Event, samples: list[float]) -> None:
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            await client.get("/health")
            samples.append(time.perf_counter() - t0)
        except Exception:
            samples.append(float("inf"))
        await asyncio.sleep(0.5)


def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"p50={statistics.median(values):.2f}s max={max(values):.2f}s"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--board", required=True)
    ap.add_argument("--class-level", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--chapter-id", required=True)
    ap.add_argument("--chapter-name", required=True)
    ap.add_argument("--query", default="Explain the main idea of this chapter in simple words.")
    a = ap.parse_args()

    async with httpx.AsyncClient(base_url=a.base, timeout=300.0) as client:
        login = await client.post("/auth/login", json={"email": a.email, "password": a.password})
        login.raise_for_status()
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        body = {
            "query": a.query,
            "board": a.board,
            "class_level": a.class_level,
            "subject_name": a.subject,
            "chapter_ids": [a.chapter_id],
            "chapter": a.chapter_name,
            "chapter_names": [a.chapter_name],
        }

        health: list[float] = []
        stop = asyncio.Event()
        poller = asyncio.create_task(poll_health(client, stop, health))
        results = await asyncio.gather(*(one_chat(client, body, headers) for _ in range(a.n)))
        stop.set()
        await poller

    errors = [r["error"] for r in results if r["error"]]
    print(f"chats: {a.n}  ok: {a.n - len(errors)}  errors: {len(errors)}")
    print("first status :", _fmt([r["status"] for r in results if r["status"] is not None]))
    print("first token  :", _fmt([r["first_token"] for r in results if r["first_token"] is not None]))
    print("total        :", _fmt([r["total"] for r in results if not r["error"]]))
    print("/health      :", _fmt(health), f"({len(health)} samples)")
    for e in errors[:5]:
        print("  error:", e)


if __name__ == "__main__":
    asyncio.run(main())

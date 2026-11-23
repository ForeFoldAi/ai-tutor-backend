"""Student-understanding eval — does the student actually LEARN the concept?

The keyword baseline (suneel_qa_baseline.py) checks whether an answer mentions the right words.
This script checks what matters for a tutor: after talking to it, does a Class-8 student
understand the concept well enough to answer a question they have not seen?

For every concept, on both channels (AI Tutor text, AI Voice = voice_mode stream):

  Turn 1  student asks about the concept (casual, Class-8 wording)
  Turn 2  student says they're still confused ("explain it more simply")
  Turn 3  student asks to be checked ("can you check if i got it?")

Then three independent measurements:

  1. Teaching rubric (LLM judge, 1-5 each, per tutor turn):
       correct      factually right for the chapter / Class 8 level
       clear        a 13-year-old can follow it (plain words, short sentences, defines terms)
       example      uses a concrete example, analogy or worked step
       builds       turn 2 explains DIFFERENTLY (simpler / new angle), not a repeat
       checks       ends by checking understanding (question, mini task)
       focused      answers what was asked, right length (voice: short and speakable)
  2. Comprehension transfer: a simulated student who knows ONLY what the tutor said answers
     a fresh quiz question; a grader marks it against the answer key (0 / 1 / 2).
     This is the headline number: "would the student understand?"
  3. Readability: Flesch-Kincaid grade of what the student reads/hears (target <= 8).

Run (FastAPI on :8000 must be up; uses the Suneel demo login like the baseline):
  cd ai-tutor-backend
  .venv/bin/python scripts/student_understanding_eval.py                 # all concepts, both channels
  .venv/bin/python scripts/student_understanding_eval.py --limit 2       # smoke test
  .venv/bin/python scripts/student_understanding_eval.py --channels voice --ids M1,So1

Writes evaluation/reports/student_understanding.{md,json} (override with --out).

Caveat: the judge and simulated student use the same LLM provider as the tutor, so absolute
scores lean generous; compare runs against each other, and read the per-turn notes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics as st
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import suneel_qa_baseline as base  # noqa: E402  (login + chat channel helpers)

OUT_DIR = ROOT / "evaluation" / "reports"
RUBRIC = ("correct", "clear", "example", "builds", "checks", "focused")
PACE_SEC = 0.0  # --pace: pause after each tutor turn (judge calls share the same AI quota)


@dataclass
class Concept:
    id: str
    subject: str
    chapter_id: str
    chapter: str
    ask: str                      # turn 1 — how a Class-8 student would ask
    key_points: list[str]         # what a good explanation must get across
    quiz: str                     # fresh transfer question (not asked in the chat)
    answer_key: str               # what a correct student answer contains
    misconception: str = ""       # common wrong idea the explanation should prevent
    confused: str = "i still don't really get it. can you explain it more simply?"
    check: str = "ok can you check if i understood it?"
    board: str = "CBSE"
    class_level: str = "CLASS_8"
    tags: list[str] = field(default_factory=list)


BANK: list[Concept] = [
    # Mathematics
    Concept(
        "M1", "Mathematics", "38", "Chapter-1 : A SQUARE AND A CUBE",
        "what is a perfect square? like what makes a number a perfect square",
        ["a perfect square is a whole number multiplied by itself", "examples like 1, 4, 9, 16, 25",
         "numbers like 2 or 10 are not perfect squares"],
        "Is 49 a perfect square? Is 50 a perfect square? Explain how you know.",
        "49 is a perfect square because 7 x 7 = 49; 50 is not, because no whole number times itself gives 50 (7x7=49, 8x8=64).",
        "thinking any number multiplied by 2 (doubling) is squaring",
    ),
    Concept(
        "M2", "Mathematics", "38", "Chapter-1 : A SQUARE AND A CUBE",
        "how are cube numbers different from square numbers?",
        ["a cube number is a number multiplied by itself three times", "e.g. 2x2x2 = 8, 3x3x3 = 27",
         "square uses two factors, cube uses three"],
        "What is the cube of 4, and is 64 also a perfect square?",
        "The cube of 4 is 4 x 4 x 4 = 64; 64 is also a perfect square because 8 x 8 = 64.",
        "thinking cube of n means n x 3",
    ),
    Concept(
        "M3", "Mathematics", "39", "Chapter-2 : POWER PLAY",
        "what does 2 to the power 5 actually mean?",
        ["the exponent tells how many times the base is multiplied by itself", "2^5 = 2x2x2x2x2 = 32",
         "it is not 2 x 5"],
        "Write 3 x 3 x 3 x 3 using a power, and find its value.",
        "3 x 3 x 3 x 3 = 3^4 (3 to the power 4) = 81.",
        "thinking 2^5 means 2 x 5 = 10",
    ),
    # Science
    Concept(
        "S1", "Science", "35", "Chapter 1 - Exploring the Investigative World of Science",
        "how do scientists actually find things out? like what steps do they follow",
        ["start from curiosity / observing and asking a question", "make a guess (hypothesis / prediction)",
         "test it with an experiment or careful observation", "draw a conclusion from the evidence"],
        "Riya sees that bread grows mould faster in summer. What should she do next to investigate it like a scientist?",
        "Make a guess (e.g. warmth makes mould grow faster) and test it fairly: keep bread in warm and cool places, keep other things the same, observe and compare, then conclude.",
        "thinking science is just memorising facts",
    ),
    Concept(
        "S2", "Science", "37", "Chapter 3 - Health: The Ultimate Treasure",
        "how do diseases spread from one person to another?",
        ["many diseases are caused by germs / microbes", "they spread through air (coughs, sneezes), "
         "contaminated water/food, touch, or insects", "hygiene like hand washing breaks the spread"],
        "Why does covering your mouth when you sneeze help stop a cold from spreading?",
        "A cold is caused by germs that travel in droplets from a sneeze; covering the mouth stops the droplets carrying germs to other people.",
        "thinking diseases spread only by eating",
    ),
    Concept(
        "S3", "Science", "37", "Chapter 3 - Health: The Ultimate Treasure",
        "what does being healthy mean? is it just not being sick?",
        ["health is more than not being ill", "it includes body (physical), mind (mental) and getting on with others (social) well-being",
         "food, exercise, sleep, cleanliness keep us healthy"],
        "Aman never falls ill but feels sad and lonely all the time. Is Aman fully healthy? Why?",
        "No — health also includes mental and social well-being, not only the body, so feeling sad and lonely means he is not fully healthy.",
        "thinking health only means not having a disease",
    ),
    # Social Science
    Concept(
        "So1", "Social", "32", "Chapter 1 - Natural Resources and Their Use",
        "what's the difference between renewable and non renewable resources?",
        ["renewable resources are replaced naturally / quickly (sunlight, wind, water, forests if managed)",
         "non-renewable resources take millions of years to form and can run out (coal, petroleum)",
         "so we must use them carefully"],
        "Is coal a renewable resource? Give a reason.",
        "No. Coal is non-renewable because it takes millions of years to form, so once used up it cannot be replaced in our lifetime.",
        "thinking all natural resources will last forever",
    ),
    Concept(
        "So2", "Social", "33", "Chapter 2 - Reshaping India’s Political Map",
        "what is a political map and how is it different from a physical map?",
        ["a political map shows boundaries, states/kingdoms/countries and who rules where",
         "a physical map shows natural features like mountains, rivers, plains",
         "political maps change over time as rulers and borders change"],
        "You want to see where the borders of different kingdoms were in 1600. Which kind of map do you need and why?",
        "A political map, because it shows boundaries of kingdoms and who ruled which area (a physical map only shows natural features).",
        "thinking political maps never change",
    ),
    Concept(
        "So3", "Social", "34", "Chapter 3 - The Rise of the Marathas",
        "how did the marathas become so powerful?",
        ["Shivaji built a strong Maratha kingdom", "strong forts and a disciplined army / guerrilla tactics in hilly terrain",
         "good administration; challenged the Mughals"],
        "Name two things Shivaji did that made the Marathas strong.",
        "Any two of: built/captured strong hill forts; used fast guerrilla-style warfare suited to the hills; organised a disciplined army; set up a good administration; challenged Mughal power.",
        "thinking the Marathas rose because of foreign help",
    ),
    # English
    Concept(
        "E1", "English", "26", "Unit 1 - Wit and Wisdom",
        "what is the lesson of the tenali rama story in wit and wisdom?",
        ["Tenali Rama solved the quarrel between the king and queen with clever humour",
         "he showed the king that yawning is natural, not disrespect",
         "wit (clever thinking) can solve problems gently"],
        "How did Tenali Rama make the king realise his mistake without directly telling him he was wrong?",
        "He used a clever comparison about seeds/yawning — asking about a farmer who yawns while sowing — so the king saw for himself that yawning is natural, instead of being told off directly.",
        "thinking the story is about Tenali Rama punishing the queen",
    ),
    Concept(
        "E2", "English", "28", "Unit 3 - Mystery and Magic",
        "how did encyclopedia solve the mystery of the four strange words?",
        ["Nolan left four words: Nom Utes Sweden Hurts", "they stand for days of the week (Monday to Thursday)",
         "the missing fifth day (Friday) pointed to where the jewels were hidden"],
        "Why did Encyclopedia think the words were connected to the days of the week?",
        "Because each strange word sounded like / matched the start of a weekday (Mon, Tues, Wednes, Thurs), and the missing next day (Friday) was the clue to the hiding place.",
        "thinking the words were in a foreign language",
    ),
    Concept(
        "E3", "English", "27", "Unit 2 - Values and Dispositions",
        "what values does the unit values and dispositions teach us?",
        ["duty, honour and patriotism through Major Somnath Sharma's sacrifice",
         "kindness and empathy (helping others)", "values guide our choices in daily life"],
        "Give one example from your own school life of showing the value of empathy.",
        "Any concrete act of understanding and helping someone because you feel what they feel, e.g. helping a classmate who is upset or struggling, linked to the unit's idea of empathy/kindness.",
        "thinking values are just rules to memorise",
    ),
]


# --------------------------------------------------------------------------- LLM helpers

_LOOP = asyncio.new_event_loop()  # one loop: llm_client keeps loop-bound clients/semaphores


def _llm(messages: list[dict], max_tokens: int = 700) -> str:
    from app.services import llm_client

    for attempt in range(3):
        try:
            return _LOOP.run_until_complete(
                llm_client.complete(messages, feature="chat", max_tokens=max_tokens, temperature=0.0)
            )
        except Exception as exc:  # provider hiccup — back off and retry
            if attempt == 2:
                raise
            print(f"   judge retry ({type(exc).__name__}: {exc})", flush=True)
            time.sleep(2 * (attempt + 1))
    return ""


def _json_from(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


_JUDGE_SYSTEM = """You are an experienced Class 8 (CBSE, India) teacher reviewing an AI tutor.
Score how well the tutor's reply helps a 13-year-old UNDERSTAND the concept. Be strict: a 5 means
you would be happy to show it in a teacher-training session.
Return ONLY JSON: {"correct":1-5,"clear":1-5,"example":1-5,"builds":1-5,"checks":1-5,"focused":1-5,
"covered_points":[indexes of key points that were clearly taught],"misconception_risk":true|false,
"note":"one short sentence on the biggest weakness"}
Definitions:
- correct: facts are right for the chapter and level; no invented facts; no wrong "not in your chapter" claims.
- clear: plain words, short sentences, new terms explained; a Class 8 student can follow it.
- example: uses a concrete example, analogy, or worked step that makes the idea click.
- builds: on a follow-up turn, it explains in a NEW/simpler way instead of repeating; on turn 1 score
  whether it starts from what a student already knows.
- checks: ends by checking understanding (a question or small task), or on a "check me" turn
  actually asks a good diagnostic question / gives clear feedback.
- focused: answers exactly what was asked at a suitable length{voice_rule}.
- misconception_risk: true if the reply could leave or create the misconception given."""


def judge_turn(c: Concept, channel: str, turn_no: int, student_msg: str, history: list[dict], reply: str) -> dict:
    voice_rule = (
        "; this is SPOKEN voice output — penalise markdown, LaTeX, lists, or replies over ~90 words"
        if channel == "voice" else ""
    )
    convo = "\n".join(f"{t['role'].upper()}: {t['content'][:900]}" for t in history[-4:])
    user = (
        f"Subject: {c.subject} | Chapter: {c.chapter}\n"
        f"Key points the explanation should get across:\n"
        + "\n".join(f"{i}. {p}" for i, p in enumerate(c.key_points))
        + f"\nMisconception to avoid: {c.misconception or 'n/a'}\n"
        f"Turn {turn_no} of 3.\n\nEarlier conversation:\n{convo or '(none)'}\n\n"
        f"STUDENT: {student_msg}\nTUTOR REPLY TO SCORE:\n{reply[:2500]}"
    )
    data = _json_from(_llm([
        {"role": "system", "content": _JUDGE_SYSTEM.replace("{voice_rule}", voice_rule)},
        {"role": "user", "content": user},
    ]))
    out = {k: _clamp(data.get(k)) for k in RUBRIC}
    out["covered_points"] = [i for i in (data.get("covered_points") or []) if isinstance(i, int)]
    out["misconception_risk"] = bool(data.get("misconception_risk"))
    out["note"] = str(data.get("note") or "")[:200]
    out["judge_ok"] = bool(data)
    return out


def _clamp(v) -> float | None:
    try:
        return max(1.0, min(5.0, float(v)))
    except (TypeError, ValueError):
        return None


_STUDENT_SYSTEM = """You are role-playing an average Class 8 student in India who had NEVER studied this
topic before today. Everything you know about it comes ONLY from the tutor's explanation below —
do not use any other knowledge. If the explanation did not teach you something, say you are not
sure or guess like a student would. Answer the quiz question in 1-3 simple sentences."""

_GRADER_SYSTEM = """You grade a Class 8 student's answer against an answer key.
The student may ONLY use what the tutor taught. Give credit only for the parts of the answer
that the tutor's explanation actually supports — if the tutor did not teach the idea (for example
it refused, said it could not find it, or talked about something else), the student could not have
learned it, so score 0 even if the answer happens to be right.
Return ONLY JSON: {"score":0|1|2,"supported_by_tutor":true|false,"reason":"short"} —
2 = correct, shows understanding, and supported by the tutor's explanation;
1 = partly right / right answer with wrong or missing reason;
0 = wrong, "not sure", or not supported by what the tutor taught."""


def comprehension_test(c: Concept, tutor_replies: list[str]) -> dict:
    taught = "\n\n".join(f"Tutor said:\n{r[:2000]}" for r in tutor_replies if r.strip())
    student = _llm([
        {"role": "system", "content": _STUDENT_SYSTEM},
        {"role": "user", "content": f"{taught or 'Tutor said nothing useful.'}\n\nQuiz question: {c.quiz}"},
    ], max_tokens=200).strip()
    grade = _json_from(_llm([
        {"role": "system", "content": _GRADER_SYSTEM},
        {"role": "user", "content": (
            f"What the tutor taught:\n{taught[:3500] or '(nothing)'}\n\n"
            f"Question: {c.quiz}\nAnswer key: {c.answer_key}\nStudent answer: {student}"
        )},
    ], max_tokens=160))
    try:
        score = int(grade.get("score"))
    except (TypeError, ValueError):
        score = 0
    if grade.get("supported_by_tutor") is False:
        score = 0
    return {"student_answer": student[:400], "score": max(0, min(2, score)),
            "supported_by_tutor": grade.get("supported_by_tutor"),
            "reason": str(grade.get("reason") or "")[:200]}


# --------------------------------------------------------------------------- readability

_MD_RE = re.compile(r"\$[^$]*\$|[*_#`>|]+|\\[a-z]+|\[[^\]]*\]\([^)]*\)")


def fk_grade(text: str) -> float:
    t = _MD_RE.sub(" ", text or "")
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", t)))
    words = re.findall(r"[A-Za-z']+", t)
    if not words:
        return 0.0
    syll = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())) - (1 if w.lower().endswith("e") and len(w) > 3 else 0))
               for w in words)
    return round(0.39 * len(words) / sentences + 11.8 * syll / len(words) - 15.59, 1)


# --------------------------------------------------------------------------- runner

def _as_item(c: Concept) -> base.QItem:
    return base.QItem(c.id, c.subject, c.chapter_id, c.chapter, c.class_level, c.board,
                      c.ask, c.confused, "normal", [])


def run_concept(token_box: dict, c: Concept, channel: str) -> dict:
    runner = base.chat_tutor if channel == "tutor" else base.chat_voice_stream
    item = _as_item(c)
    history: list[dict] = []
    turns = []
    for n, msg in enumerate((c.ask, c.confused, c.check), 1):
        res = _call(runner, token_box, item, msg, history)
        reply = (res.get("answer") or "").strip()
        if channel == "voice" and reply:
            # Judge what the student HEARS — markdown is stripped before speech.
            from app.services.tts_sanitize import sanitize_for_tts

            reply = sanitize_for_tts(reply) or reply
        j = judge_turn(c, channel, n, msg, history, reply) if reply else {
            **{k: 1.0 for k in RUBRIC}, "covered_points": [], "misconception_risk": True,
            "note": f"no reply ({res.get('error') or 'empty'})", "judge_ok": False,
        }
        turns.append({
            "turn": n, "student": msg, "reply": reply, "ms": res.get("ms"), "error": res.get("error"),
            "n_images": len(res.get("related_images") or []), "fk_grade": fk_grade(reply),
            "words": len(reply.split()), "judge": j,
        })
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        if PACE_SEC:
            time.sleep(PACE_SEC)  # stay under the provider's tokens-per-minute cap
        print(f"   [{channel}] {c.id} t{n} {res.get('ms')}ms words={turns[-1]['words']} "
              f"judge={ {k: j[k] for k in RUBRIC} }", flush=True)
    comp = comprehension_test(c, [t["reply"] for t in turns[:2]])
    print(f"   [{channel}] {c.id} comprehension={comp['score']}/2 — {comp['reason']}", flush=True)
    covered = set().union(*(set(t["judge"]["covered_points"]) for t in turns))
    return {
        "id": c.id, "subject": c.subject, "chapter": c.chapter, "channel": channel,
        "turns": turns, "comprehension": comp,
        "key_point_coverage": round(len(covered & set(range(len(c.key_points)))) / len(c.key_points), 2),
        "misconception_risk": any(t["judge"]["misconception_risk"] for t in turns),
        "understanding_score": understanding_score(turns, comp, covered, c),
    }


def _call(runner, token_box, item, msg, history):
    for attempt in range(2):
        try:
            res = runner(token_box["t"], item, msg, history)
        except Exception as exc:
            res = {"answer": "", "error": str(exc)[:200], "ms": 0}
        if "401" in str(res.get("error") or "") and attempt == 0:
            token_box["t"] = base.login()
            continue
        return res
    return res


def understanding_score(turns: list[dict], comp: dict, covered: set[int], c: Concept) -> float:
    """0-10 headline: 40% comprehension transfer, 40% teaching rubric, 20% key-point coverage."""
    rub = [t["judge"][k] for t in turns for k in RUBRIC if t["judge"].get(k) is not None]
    rubric10 = (st.mean(rub) - 1) / 4 * 10 if rub else 0
    comp10 = comp["score"] / 2 * 10
    cov10 = len(covered & set(range(len(c.key_points)))) / len(c.key_points) * 10
    return round(0.4 * comp10 + 0.4 * rubric10 + 0.2 * cov10, 2)


# --------------------------------------------------------------------------- report

def summarize(rows: list[dict]) -> dict:
    def m(vals):
        vals = [v for v in vals if v is not None]
        return round(st.mean(vals), 2) if vals else None

    out = {}
    for ch in sorted({r["channel"] for r in rows}):
        rs = [r for r in rows if r["channel"] == ch]
        tts = [t for r in rs for t in r["turns"]]
        out[ch] = {
            "understanding_score": m([r["understanding_score"] for r in rs]),
            "comprehension_pct": round(100 * sum(r["comprehension"]["score"] for r in rs) / (2 * len(rs)), 1),
            "key_point_coverage_pct": round(100 * m([r["key_point_coverage"] for r in rs]), 1),
            "misconception_risk_items": sum(r["misconception_risk"] for r in rs),
        "busy_fallback_turns": sum(
            1 for t in tts if "ask me again" in t["reply"].lower() or "ask again" in t["reply"].lower()
        ),
            "rubric": {k: m([t["judge"][k] for t in tts]) for k in RUBRIC},
            "rubric_by_turn": {
                n: {k: m([t["judge"][k] for t in tts if t["turn"] == n]) for k in RUBRIC} for n in (1, 2, 3)
            },
            "fk_grade_avg": m([t["fk_grade"] for t in tts if t["reply"]]),
            "words_avg": m([t["words"] for t in tts]),
            "latency_ms_avg": m([t["ms"] for t in tts if t["ms"]]),
            "latency_ms_p90": sorted(t["ms"] or 0 for t in tts)[int(0.9 * (len(tts) - 1))] if tts else None,
            "failed_turns": sum(1 for t in tts if not t["reply"] or t["error"]),
            "by_subject": {s: m([r["understanding_score"] for r in rs if r["subject"] == s])
                           for s in sorted({r["subject"] for r in rs})},
        }
    return out


def write_md(path: Path, payload: dict) -> None:
    S = payload["summary"]
    chans = list(S)
    L = ["# Student Understanding Eval", "",
         f"- **Date:** {payload['generated_at']}",
         f"- **Concepts:** {len({r['id'] for r in payload['rows']})} · 3 turns each "
         "(ask → \"still confused, simpler please\" → \"check if I got it\")",
         "- **Headline (0-10):** 40% simulated-student quiz on a NEW question + 40% teacher rubric + 20% key-point coverage",
         "- **Judge caveat:** same LLM provider as the tutor — compare runs, don't treat absolutes as exact.", "",
         "## Ranking", "",
         "| Rank | Channel | Understanding /10 | Student quiz correct | Key points taught | Misconception risk | Reading grade | Avg words | Avg latency |",
         "|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(chans, key=lambda c: -(S[c]["understanding_score"] or 0))
    for i, ch in enumerate(ranked, 1):
        s = S[ch]
        L.append(f"| {i} | {'AI Tutor (text)' if ch == 'tutor' else 'AI Voice'} | **{s['understanding_score']}** | "
                 f"{s['comprehension_pct']}% | {s['key_point_coverage_pct']}% | {s['misconception_risk_items']} items | "
                 f"{s['fk_grade_avg']} | {s['words_avg']} | {round((s['latency_ms_avg'] or 0) / 1000, 1)} s |")
    L += ["", "## Teaching rubric (1-5, all turns)", "",
          "| Dimension | " + " | ".join(chans) + " |", "|---|" + "---|" * len(chans)]
    for k in RUBRIC:
        L.append(f"| {k} | " + " | ".join(str(S[c]["rubric"][k]) for c in chans) + " |")
    L += ["", "### By turn", ""]
    for ch in chans:
        L.append(f"**{ch}**")
        L.append("")
        L.append("| Turn | " + " | ".join(RUBRIC) + " |")
        L.append("|---|" + "---|" * len(RUBRIC))
        for n, d in S[ch]["rubric_by_turn"].items():
            L.append(f"| {n} | " + " | ".join(str(d[k]) for k in RUBRIC) + " |")
        L.append("")
    L += ["## By subject (understanding /10)", "", "| Subject | " + " | ".join(chans) + " |", "|---|" + "---|" * len(chans)]
    for subj in sorted({r["subject"] for r in payload["rows"]}):
        L.append(f"| {subj} | " + " | ".join(str(S[c]["by_subject"].get(subj)) for c in chans) + " |")
    L += ["", "## Per concept", ""]
    for cid in dict.fromkeys(r["id"] for r in payload["rows"]):
        rs = [r for r in payload["rows"] if r["id"] == cid]
        L.append(f"### {cid} — {rs[0]['subject']} · {rs[0]['chapter']}")
        L.append("")
        for r in rs:
            comp = r["comprehension"]
            L.append(f"- **{r['channel']}** — understanding **{r['understanding_score']}**, quiz {comp['score']}/2, "
                     f"key points {int(r['key_point_coverage'] * 100)}%, misconception risk {r['misconception_risk']}")
            L.append(f"  - Simulated student: _{comp['student_answer'][:220]}_ — grader: {comp['reason']}")
            for t in r["turns"]:
                j = t["judge"]
                L.append(f"  - T{t['turn']} ({t['words']} words, grade {t['fk_grade']}, {t['ms']} ms): "
                         + ", ".join(f"{k} {j[k]}" for k in RUBRIC) + (f" — _{j['note']}_" if j.get("note") else ""))
                L.append(f"    > {t['reply'][:260].replace(chr(10), ' ')}{'…' if len(t['reply']) > 260 else ''}")
        L.append("")
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channels", default="tutor,voice")
    ap.add_argument("--ids", default="", help="comma list of concept ids (default all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="student_understanding")
    ap.add_argument("--pace", type=float, default=20.0,
                    help="seconds to pause after each tutor turn (default 20; 0 to disable)")
    a = ap.parse_args()
    global PACE_SEC
    PACE_SEC = a.pace

    concepts = [c for c in BANK if not a.ids or c.id in a.ids.split(",")]
    if a.limit:
        concepts = concepts[: a.limit]
    token_box = {"t": base.login()}
    rows = []
    for ch in a.channels.split(","):
        print(f"=== {ch} ===", flush=True)
        for c in concepts:
            rows.append(run_concept(token_box, c, ch))
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "rows": rows, "summary": summarize(rows)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{a.out}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_md(OUT_DIR / f"{a.out}.md", payload)
    print(json.dumps(payload["summary"], indent=1))
    print("Wrote", OUT_DIR / f"{a.out}.md")


if __name__ == "__main__":
    main()

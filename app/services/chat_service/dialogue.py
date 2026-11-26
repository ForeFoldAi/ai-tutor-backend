"""Dialogue-act helpers for greetings, affirmations, and continue offers."""
from __future__ import annotations

import re
from typing import Any

logger = __import__("logging").getLogger(__name__)


_AFFIRMATION_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|ok|okay|sure|right|correct|exactly|got\s+it|"
    r"(?:i\s+)?(?:understand|understood)|makes\s+sense|that\s+helps|"
    r"clear\s+now|sounds\s+good|"
    # "Good thanks" / "thanks" / "thank you" mid-lesson = understood, not a fresh greeting
    r"(?:good|great|cool|nice|awesome)?\s*(?:thanks|thank\s+you|thx)|"
    r"appreciate\s+(?:it|that)|that'?s\s+(?:helpful|great|good)"
    r")(?:\s+.*)?[.!?]*$",
    re.I,
)


_EXPAND_REQUEST_RE = re.compile(
    r"\b(?:"
    r"explain(?:\s+more|\s+please)?|please\s+explain|explore|elaborate|"
    r"tell\s+me(?:\s+more)?|go\s+(?:on|ahead)|continue|carry\s+on|"
    r"next(?:\s+part)?|deeper|more\s+(?:about|detail|on)|"
    r"how\s+(?:it|they|one)\s+affects?"
    r")\b",
    re.I,
)


_TUTOR_CONTINUE_OFFER_RE = re.compile(
    r"would you like to\s+(?:explore|learn|look|see|try|continue|explain|"
    r"dive|understand|know|hear|discuss|go)|"
    r"want (?:me )?to\s+(?:explain|explore|show|teach|continue)|"
    r"shall (?:we|i)\s+(?:explore|look|continue|explain)",
    re.I,
)


_MULTI_CHOICE_MENU_RE = re.compile(
    r"(?:real[- ]?life\s+example|quick\s+quiz|explore\s+other\s+topics|"
    r"example,\s*(?:a\s+)?(?:quick\s+)?quiz|,\s*or\s+(?:to\s+learn|explore))",
    re.I,
)


_PERSONAL_EXAMPLE_RE = re.compile(
    r"\b(i was|it was|when i|where i|one day|one time|suddenly|yesterday|last\s+(?:week|month|year)|"
    r"started\s+(?:rain|snow)|raining|sunny|cloudy|snowing|storm)\b",
    re.I,
)


_SESSION_GREETING_RE = re.compile(
    r"(?:"
    r"\b(?:welcome back|i(?:'m| am) your ai tutor)\b.*\bwhat would you like to learn\b"
    r"|"
    r"\bhere are a few things you can ask me\b"
    r"|"
    r"\bpick one, or type your own question\b"
    r")",
    re.I | re.S,
)


_ACADEMIC_QUESTION_RE = re.compile(
    r"\b("
    r"how (?:far|long|many|much)|can (?:you|we|i) reach|what is|what are|explain|define|"
    r"solve|calculate|find|prove|distance|travel|moon|speed|rate|per day|per hour|"
    r"equation|formula|\d+\s*(?:km|m|cm|mm|years?|days?|hours?|minutes?|%)"
    r")\b",
    re.I,
)


def _is_session_greeting_message(text: str) -> bool:
    return bool(_SESSION_GREETING_RE.search((text or "").strip()))


def _last_assistant_text(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return ""
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "assistant":
            return (turn.get("content") or "").strip()
    return ""


def _is_personal_dialogue_response(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """Student answered the tutor's question with an example or personal experience."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if _ACADEMIC_QUESTION_RE.search(q):
        return False
    last_asst = _last_assistant_text(conversation_history)
    if not last_asst or "?" not in last_asst:
        return False
    if _is_session_greeting_message(last_asst):
        return False
    if _PERSONAL_EXAMPLE_RE.search(q):
        return True
    words = q.split()
    if "," in q and len(words) >= 5 and _PERSONAL_EXAMPLE_RE.search(q):
        return True
    return False


def _last_assistant_offered_to_continue(
    conversation_history: list[dict] | None,
) -> bool:
    """True when the tutor asked a single yes/no-style 'shall we explore…?' offer."""
    last = _last_assistant_text(conversation_history)
    if not last or "?" not in last:
        return False
    if _MULTI_CHOICE_MENU_RE.search(last):
        return False
    return bool(_TUTOR_CONTINUE_OFFER_RE.search(last))


def _is_accepting_tutor_continue_offer(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """Student accepted the tutor's offer to keep explaining (e.g. 'Yes please explain')."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if _EXPAND_REQUEST_RE.search(q):
        return bool(_last_assistant_text(conversation_history))
    if not _last_assistant_offered_to_continue(conversation_history):
        return False
    if not _AFFIRMATION_RE.match(q):
        return False
    return len(q.split()) <= 7


def _continue_offer_prompt_override(
    query: str, conversation_history: list[dict] | None
) -> dict[str, str] | None:
    """Prompt bits when the student said yes to a specific tutor offer (not a choice menu)."""
    if not _is_accepting_tutor_continue_offer(query, conversation_history):
        return None
    last = _last_assistant_text(conversation_history)
    offer = ""
    if last and "?" in last:
        # Prefer the last question the tutor asked — that is what "yes" accepts.
        parts = re.split(r"(?<=[?])\s*", last)
        for part in reversed(parts):
            s = (part or "").strip()
            if s.endswith("?") and _TUTOR_CONTINUE_OFFER_RE.search(s):
                offer = s
                break
        if not offer:
            offer = parts[-1].strip() if parts else ""
    offer_line = f' Their yes accepts: "{offer}"' if offer else ""
    return {
        "instruction": (
            "The student accepted YOUR specific yes/no offer from the previous message."
            f"{offer_line} "
            "Teach that exact next step now using the chapter context. "
            "Do NOT open a choice menu (real-life example / quick quiz / other topics / wrap up). "
            "Do NOT ask again whether they want that topic."
        ),
        "length_policy": (
            "RESPONSE LENGTH: about 80-140 words. Plain prose — no multi-choice menu."
        ),
        "interaction_policy": (
            "End with ONE short check question on what you just taught — "
            "not a quiz offer and not a menu of options."
        ),
        "explanation_structure": "",
        "user_closing": (
            "The student said yes to your offer. Deliver that next explanation now "
            "(the topic you just offered) — do not offer a quiz or choice menu:"
        ),
    }


def _struggle_simplify_prompt_override(query: str) -> dict[str, str] | None:
    """When the student asks to simplify / says they're confused — short plain teaching."""
    from app.services.conversation_intent_classifier import FollowupType, classify_followup_regex

    q = (query or "").strip()
    if not q:
        return None
    act = classify_followup_regex(q)
    explicit = bool(
        re.search(
            r"\b(confus(?:ed|ing)|explain\s+like|i\s+keep\s+forgetting|still\s+stuck|"
            r"like\s+i(?:'?m|\s+am)\s+(?:dumb|stupid)|keep\s+it\s+simple|in\s+simple\s+words)\b",
            q,
            re.I,
        )
    )
    if act not in (FollowupType.SIMPLIFY, FollowupType.CLARIFICATION) and not explicit:
        return None
    return {
        "instruction": (
            "The student is struggling or asked for a simpler explanation. "
            "Open with one warm line (\"It's okay to be stuck\"). "
            "Use everyday words only — no Snell's law, no sin i / sin r, no long formula dump "
            "unless they asked for the formula. Never say \"according to the chapter\". "
            "One analogy. Max 3 short steps. "
            "About 60–110 words. Do NOT append science-experiment JSON. "
            "End with ONE yes/no check."
        ),
        "length_policy": "RESPONSE LENGTH: 60–110 words. Short sentences under 14 words.",
        "interaction_policy": (
            "Warm + tiny steps + one check question. No quiz menu. No experiment JSON."
        ),
        "explanation_structure": "",
        "user_closing": (
            "Write a simple, kind explanation now (no jargon dump, no experiment JSON):"
        ),
    }


def _chapter_meta_prompt_override(
    query: str, overview_part: tuple[int, int] | None = None
) -> dict[str, str] | None:
    """Prompt bits for topic-list / whole-chapter asks (text chat).

    ``overview_part`` = (part, parts) when a long chapter is taught in parts.
    """
    from app.modules.student_learning.topic_progress import is_chapter_topics_list_query
    from app.services.voice_chapter_vocab import is_chapter_overview_request

    q = (query or "").strip()
    if not q:
        return None
    if is_chapter_topics_list_query(q):
        return {
            "instruction": (
                "The student wants the FULL topic list for this chapter. "
                "List EVERY main topic from the CHAPTER TOPICS list / chapter context in order "
                "as a numbered list (include section numbers like 9.1 when present). "
                "Do NOT teach the content yet. Do NOT open a quiz/example menu. "
                "Do NOT invent topics missing from the context."
            ),
            "length_policy": "RESPONSE LENGTH: as long as the topic list needs — plain numbered list.",
            "interaction_policy": (
                "After the list, ask ONE short question: which topic to start with. "
                "No quiz offer, no multi-choice menu."
            ),
            "explanation_structure": "",
            "user_closing": (
                "Write the complete numbered topic list now, then ask which topic to start with:"
            ),
        }
    from app.services.chapter_scope import is_gap_check_query

    if is_gap_check_query(q):
        return {
            "instruction": (
                "The student is asking what your previous answer left out of this chapter. "
                "Read the CHAPTER TOPICS list and your previous answer. "
                "Explain only the listed topics that previous answer did not already teach, "
                "in order, with a short bold heading and 1–2 sentences each. "
                "If every listed topic is already in that answer, say you did not miss anything "
                "and stop. Do not repeat covered topics. Do not invent topics. "
                "Place names, map labels, quotations, and pieces of the chapter title are not topics. "
                "Do not attach a science-experiment block or a quiz menu."
            ),
            "length_policy": (
                "RESPONSE LENGTH: only as long as the still-missing topics need. "
                "If nothing is missing, one sentence."
            ),
            "interaction_policy": "Do not end with a simulation offer, quiz, or choice menu.",
            "explanation_structure": "",
            "user_closing": (
                "Write only the missing topics now, or say you did not miss anything:"
            ),
        }
    if is_chapter_overview_request(q) and overview_part and overview_part[1] > 1:
        part, parts = overview_part
        last = part >= parts
        return {
            "instruction": (
                f"The student wants the chapter explained. It is taught in {parts} parts; "
                f"this is PART {part} OF {parts}. "
                "Walk through ONLY the topics in the CHAPTER TOPICS list, in order — the "
                "other topics are covered in the other parts, so do not mention or summarise them. "
                "For each topic use a short bold heading (keep the number shown in the list) and "
                "2–3 sentences of what it covers. "
                "Do NOT invent facts outside the context. "
                "Do NOT number a place name, a map label, a quotation, a figure caption, "
                "or a fragment of the chapter title. Do NOT leave an empty numbered item. "
                "Do NOT attach a science-experiment block. "
                "Do NOT open a quiz/example choice menu."
            ),
            "length_policy": (
                "RESPONSE LENGTH: one short block per listed topic, in order "
                "(about 50–80 words each)."
            ),
            "interaction_policy": (
                "This is the last part: end with ONE invite to dive deeper into any topic of the chapter."
                if last
                else f"End with ONE short line telling the student to press Continue for part {part + 1} "
                "or to ask about any of these topics first."
            ),
            "explanation_structure": "",
            "user_closing": f"Write part {part} of the chapter walkthrough now (only the listed topics):",
        }
    if is_chapter_overview_request(q):
        return {
            "instruction": (
                "The student wants the WHOLE chapter explained. "
                "Walk through EVERY main topic from the CHAPTER TOPICS list / chapter context "
                "in order. For each topic use a short bold heading (section number + title when "
                "present) and 1–2 sentences of what it covers. Do not stop after 2–3 topics. "
                "Do NOT invent facts outside the context. "
                "Do NOT number a place name, a map label, a quotation, a figure caption, "
                "or a fragment of the chapter title. Do NOT leave an empty numbered item. "
                "Do NOT attach a science-experiment block. "
                "Do NOT open a quiz/example choice menu."
            ),
            "length_policy": (
                "RESPONSE LENGTH: one short block per listed topic, in order "
                "(about 40–70 words each). Do not stop after the first two topics."
            ),
            "interaction_policy": (
                "End with ONE invite to dive deeper into the first topic — "
                "not a quiz menu and not multiple options."
            ),
            "explanation_structure": "",
            "user_closing": (
                "Write the full ordered chapter walkthrough now (every topic), "
                "then invite them to start with the first topic:"
            ),
        }
    return None


def _is_affirmation_followup(query: str, conversation_history: list[dict] | None) -> bool:
    """True when the student gives a short bare acknowledgment (not a story or answer)."""
    if not conversation_history:
        return False
    q = (query or "").strip()
    if not q or not _AFFIRMATION_RE.match(q):
        return False
    if not _last_assistant_text(conversation_history):
        return False
    # "Yes please explain" / accepting an explore offer → keep teaching
    if _EXPAND_REQUEST_RE.search(q):
        return False
    if _is_accepting_tutor_continue_offer(q, conversation_history):
        return False
    words = q.split()
    if len(words) > 7:
        return False
    if "," in q and len(words) > 4:
        return False
    if _PERSONAL_EXAMPLE_RE.search(q):
        return False
    return True


def _skip_answer_expansion(
    query: str, conversation_history: list[dict] | None
) -> bool:
    from app.services.conversation_context import is_clarification_followup

    return (
        _is_affirmation_followup(query, conversation_history)
        or _is_personal_dialogue_response(query, conversation_history)
        or is_clarification_followup(query, conversation_history)
    )


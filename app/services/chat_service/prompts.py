"""Chat prompt templates and message builders."""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.prompts import PromptTemplate

from app.services.chat_service.answer_types import (
    _ANSWER_INSTRUCTIONS,
    _ANSWER_MIN_WORDS,
    _MATH_DIALOGUE_TYPES,
    _MATH_SHORT_TYPES,
    _QUIZ_TYPES,
    _SUBJECT_GUIDELINES_MATH_CONCEPT,
    _SUBJECT_GUIDELINES_MATH_ELEMENTARY,
    _apply_agent_mode,
    _build_question_type_guidance,
    _build_subject_guidelines,
    _class_band,
    _is_mathematics_subject,
    _is_science_subject,
    _math_format_tier,
    _normalize_agent_mode,
    _resolve_answer_type,
    _should_use_elementary_math_format,
    _structure_tier,
    detect_answer_type,
    detect_question_type,
)
from app.services.chat_service.dialogue import (
    _is_affirmation_followup,
    _is_personal_dialogue_response,
    _skip_answer_expansion,
)
from app.services.chat_service.postprocess import _DIRECT_ANSWER_MAX_WORDS

logger = logging.getLogger(__name__)


_EXPLANATION_STRUCTURE_MATH = """\
MATHEMATICS ANSWER FORMAT (required for every mathematics question):
Use these sections in order. Each section title MUST be on its own line wrapped in **bold** \
(e.g. **To Find**). NO # symbols, NO emoji.

**To Find**
Briefly explain what the question is asking.

**Given Information**
List all numbers, values, units, and important details provided in the question.

**Concept Behind It**
Explain which mathematical concept is being used, why it applies, and any rule or property involved.

**The Formula**
Present every formula in BOTH forms (mandatory when a formula applies). Never skip either form. \
Never write the word "or" on its own line.
Before using any letter (R, P, A, B, t, …), name what it means in words.

Example (division):
Number of buses = Total people ÷ People per bus
$$\\text{Number of buses} = \\frac{\\text{Total people}}{\\text{People per bus}}$$

Example (multiplication):
Total Distance = daily distance × number of days
$$\\text{Total Distance} = \\text{daily distance} \\times \\text{number of days}$$

Rules:
- Line 1: word/symbol form using ÷, ×, +, − as needed.
- Line 2: display LaTeX ($$...$$) — use \\frac{}{} for division, \\times for multiplication.
- Do NOT put "or" between the two lines.
- Use \\text{...} inside LaTeX for words. Use Indian number grouping in prose (e.g. 12,00,000).
- If you use a short letter, define it once in the same section (e.g. \"R = puffing rate\").

**Solution**
Substituting the values:

Then show each calculation on its own line as display LaTeX, starting with =:
$$= 5 \\times 365$$
$$= 1{,}825$$
$$= 200 \\times 1{,}825$$
$$= 3{,}65{,}000$$

Rules:
- Each substitution / simplification step gets its own $$...$$ line starting with =.
- Never skip arithmetic steps between substitution and the final value.
- Number steps (Step 1, Step 2, …) only when extra reasoning is needed before substituting.

**Quick Check**
Verify the answer using estimation, reverse calculation, or logical reasoning whenever possible.

**Final Answer**
Clearly present the final answer in **bold** (e.g. **No, you cannot reach the Moon.**).

**Key Takeaway**
Summarize the main mathematical idea learned from this problem in 1–2 simple sentences.

**Practice Question**
Generate one similar question for the student to try independently.

The student should finish feeling: (1) they understand the concept, (2) they understand each step, \
(3) they can solve a similar problem alone."""


_LENGTH_POLICY_MATH = """\
RESPONSE LENGTH — MATHEMATICS:
- Easy questions: 100–200 words
- Medium questions: 200–400 words
- Difficult questions: 400–600 words
- Match length to problem difficulty; never pad with unrelated content.
- If chapter context is short, expand only using what the chapter supports."""


_LENGTH_POLICY_MATH_SHORT = """\
RESPONSE LENGTH — MATHEMATICS (short request):
- The student asked for a short answer. Give the final answer plus the formula (word form + fraction LaTeX) \
and 2–3 essential $$= ...$$ substitution lines (about 50–120 words).
- Do NOT use the full eight-section template or emoji section headers."""


_INTERACTION_MATH = """\
FOLLOW-UP RULES (mathematics):
- End the prose answer with the **Practice Question** section.
- AFTER **Practice Question**, you MUST append the ```math-lesson``` JSON visualization block (required — not a generic follow-up).
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""


_EXPLANATION_STRUCTURE_MATH_ELEMENTARY = """\
ELEMENTARY MATHEMATICS ANSWER FORMAT (Classes 1–5 — concept / compare / explain questions):
- Use plain, friendly paragraphs only. NO section headers like **To Find**, **Given Information**, **The Formula**.
- 2–4 short sentences per paragraph. One simple real-life example (toys, food, classroom, home).
- You may use ONE optional **Remember** line at the end — nothing else in bold as a heading.
- Skip formulas unless the student asked to calculate something.
- End with ONE fun check question in plain text (not a **Practice Question** section)."""


_LENGTH_POLICY_MATH_ELEMENTARY = """\
RESPONSE LENGTH — ELEMENTARY MATHEMATICS:
- Keep the whole answer about 60–120 words unless the student asked for more.
- Short, fun, and easy to read aloud. No exam-style structure."""


_INTERACTION_MATH_ELEMENTARY = """\
FOLLOW-UP (elementary):
- End with one short spoken-style question (e.g. "Can you spot a rectangle in your classroom?").
- You MUST append the ```math-lesson``` JSON block AFTER the prose (Interactive Exploration is mandatory)."""


_EXPLANATION_STRUCTURE_MATH_MIDDLE = """\
MATHEMATICS CONCEPT ANSWER FORMAT (Classes 6–8 — explain / compare / discuss):
- Use 2–3 clear paragraphs with correct basic terminology (define terms briefly).
- Compare similarities and differences when the student asks to compare.
- Include one relatable real-life example.
- You may use a short bullet list of key properties (3–5 bullets max).
- Do NOT use **To Find**, **Given Information**, **The Formula** exam-style headers."""


_LENGTH_POLICY_MATH_MIDDLE = """\
RESPONSE LENGTH — MATHEMATICS (Classes 6–8 concept):
- About 100–180 words. Clear and organised — not exam-essay length."""


_INTERACTION_MATH_MIDDLE = """\
FOLLOW-UP (Classes 6–8 concept):
- End with one check question tied to what you taught.
- You MUST append the ```math-lesson``` JSON block AFTER the prose."""


_EXPLANATION_STRUCTURE_MATH_SECONDARY = """\
MATHEMATICS CONCEPT ANSWER FORMAT (Classes 9–12 — explain / compare / discuss):
- Use well-structured paragraphs with proper mathematical terminology.
- State definitions, properties, and relationships between concepts clearly.
- Distinguish special cases (e.g. a square is a special type of rectangle).
- Include exam-relevant points when helpful.
- You may use a short bullet list of key properties.
- Do NOT use eight-section **To Find** / **Formula** headers unless solving a numeric problem."""


_LENGTH_POLICY_MATH_SECONDARY = """\
RESPONSE LENGTH — MATHEMATICS (Classes 9–12 concept):
- About 150–280 words. Detailed, exam-ready vocabulary in flowing prose — not section headers."""


_INTERACTION_MATH_SECONDARY = """\
FOLLOW-UP (Classes 9–12 concept):
- End with one thoughtful check question or offer to go deeper.
- You MUST append the ```math-lesson``` JSON block AFTER the prose."""


def _apply_mathematics_prompt_overrides(
    *,
    subject_name: str,
    answer_type: str,
    heading_scope: Any | None,
    class_level: str = "",
    question_type: str = "conceptual",
    instruction: str,
    length_policy: str,
    interaction_policy: str,
    explanation_structure: str,
    user_closing: str,
) -> tuple[str, str, str, str, str]:
    """Apply eight-section mathematics format when subject is Mathematics."""
    if not _is_mathematics_subject(subject_name):
        return instruction, length_policy, interaction_policy, explanation_structure, user_closing
    if answer_type in _MATH_DIALOGUE_TYPES:
        return instruction, length_policy, interaction_policy, explanation_structure, user_closing

    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return instruction, length_policy, interaction_policy, explanation_structure, user_closing

    if answer_type in _MATH_SHORT_TYPES:
        return (
            instruction,
            _LENGTH_POLICY_MATH_SHORT,
            _INTERACTION_BRIEF,
            "",
            "Write your short mathematics answer now — final answer plus 2–4 essential numbered steps "
            "(no eight-section template):",
        )

    if _should_use_elementary_math_format(class_level, question_type):
        return (
            "Explain like a kind Class 1–5 teacher. Plain paragraphs only — no eight-section template.",
            _LENGTH_POLICY_MATH_ELEMENTARY,
            _INTERACTION_MATH_ELEMENTARY,
            _EXPLANATION_STRUCTURE_MATH_ELEMENTARY,
            "Write your short, kid-friendly mathematics answer now (plain prose, then math-lesson JSON if showing):",
        )

    tier = _math_format_tier(class_level, question_type)
    if tier == "middle":
        return (
            "Explain clearly for a Class 6–8 student. Use paragraphs and basic terminology — no eight-section template.",
            _LENGTH_POLICY_MATH_MIDDLE,
            _INTERACTION_MATH_MIDDLE,
            _EXPLANATION_STRUCTURE_MATH_MIDDLE,
            "Write your Class 6–8 concept answer now (paragraphs + bullets if helpful, then math-lesson JSON):",
        )
    if tier == "secondary":
        return (
            "Explain clearly for a Class 9–12 student. Use academic terminology in flowing prose — "
            "no eight-section template for this concept question.",
            _LENGTH_POLICY_MATH_SECONDARY,
            _INTERACTION_MATH_SECONDARY,
            _EXPLANATION_STRUCTURE_MATH_SECONDARY,
            "Write your Class 9–12 concept answer now (detailed prose, then math-lesson JSON):",
        )

    math_instruction = (
        "Follow the mandatory mathematics teaching format exactly (**bold** section titles, no # or emoji). "
        "Never give only the final answer."
    )
    return (
        math_instruction,
        _LENGTH_POLICY_MATH,
        _INTERACTION_MATH,
        _EXPLANATION_STRUCTURE_MATH,
        f"Write your complete mathematics tutor answer now ({answer_type}, "
        "all sections in order: To Find through Practice Question):",
    )


_GRADE_LABELS: dict[str, str] = {
    "CLASS_1": "Class 1 (age 6-7)",
    "CLASS_2": "Class 2 (age 7-8)",
    "CLASS_3": "Class 3 (age 8-9)",
    "CLASS_4": "Class 4 (age 9-10)",
    "CLASS_5": "Class 5 (age 10-11)",
    "CLASS_6": "Class 6 (age 11-12)",
    "CLASS_7": "Class 7 (age 12-13)",
    "CLASS_8": "Class 8 (age 13-14)",
    "CLASS_9": "Class 9 (age 14-15)",
    "CLASS_10": "Class 10 (age 15-16)",
    "CLASS_11": "Class 11 (age 16-17)",
    "CLASS_12": "Class 12 (age 17-18)",
}


_GRADE_COMPLEXITY: dict[str, str] = {
    "CLASS_1": (
        "Speak like a kind, patient teacher talking to a 6-year-old. "
        "Use only the simplest words. One short idea per sentence. "
        "No jargon at all. Use fun comparisons from daily life."
    ),
    "CLASS_2": (
        "Speak like a kind teacher talking to a 7-year-old. "
        "Use only simple everyday words. One idea per sentence. "
        "No technical terms. Make it feel like a friendly story."
    ),
    "CLASS_3": (
        "Use simple, friendly language for an 8-year-old student. "
        "Short sentences. Everyday words only. "
        "Give a simple real-life example when helpful."
    ),
    "CLASS_4": (
        "Use clear, simple language for a 9-year-old. "
        "Short sentences. Introduce a basic term only if truly needed, "
        "and explain it right away in simple words."
    ),
    "CLASS_5": (
        "Use clear, friendly language for a 10-year-old. "
        "You may use basic subject terms but explain them simply. "
        "Add a helpful example to make the idea stick."
    ),
    "CLASS_6": (
        "Use clear, natural language for an 11-year-old. "
        "Subject-specific terms are fine but always explain them briefly. "
        "Keep sentences short and easy to read."
    ),
    "CLASS_7": (
        "Use clear, confident language for a 12-year-old. "
        "Use correct subject terms with a short explanation. "
        "Answers can be slightly more detailed but still easy to follow."
    ),
    "CLASS_8": (
        "Use standard, friendly academic language for a 13-year-old. "
        "Well-structured answers. Use correct terminology. "
        "Keep the tone warm and encouraging, not textbook-robotic."
    ),
    "CLASS_9": (
        "Use standard academic language for a 14-15-year-old. "
        "Use proper terminology. Answers can be detailed and well-organised. "
        "Maintain a helpful, clear, exam-ready tone."
    ),
    "CLASS_10": (
        "Use precise academic language for a 15-16-year-old board exam student. "
        "Use correct technical terms. Give thorough, well-structured answers. "
        "Tone should be clear, confident, and exam-ready."
    ),
    "CLASS_11": (
        "Provide deeper conceptual understanding for a 16-17-year-old. "
        "Use advanced terminology where appropriate. Explain reasoning and links between ideas. "
        "Prepare for competitive exams and higher studies."
    ),
    "CLASS_12": (
        "Provide deeper conceptual understanding for a 17-18-year-old. "
        "Use advanced terminology where appropriate. Explain reasoning and relationships between concepts. "
        "Prepare for board exams, competitive exams, and higher studies."
    ),
}


_CLASS_BAND_RULES: dict[str, str] = {
    "1-5": (
        "Classes 1-5: Use simple language and everyday examples. "
        "Keep explanations short and fun. Avoid technical terminology."
    ),
    "6-8": (
        "Classes 6-8: Use moderate detail. Introduce basic scientific and academic terms. "
        "Include relatable examples."
    ),
    "9-10": (
        "Classes 9-10: Provide detailed explanations and underlying concepts. "
        "Include exam-oriented points and real-world applications."
    ),
    "11-12": (
        "Classes 11-12: Provide deeper conceptual understanding. "
        "Use advanced terminology where appropriate. Explain reasoning and concept relationships."
    ),
}


def _student_first_name(full_name: str) -> str:
    name = (full_name or "").strip()
    if not name:
        return "there"
    return name.split()[0]


def build_session_greeting(
    *,
    student_name: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
) -> str:
    """Personalized welcome when a student taps Start Learning."""
    first = _student_first_name(student_name)
    subject = (subject_name or "").strip() or "your subject"
    names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    if not names and chapter:
        names = [chapter.strip()]
    if len(names) == 1:
        from_topic = names[0]
    elif len(names) == 2:
        from_topic = f"{names[0]} and {names[1]}"
    elif len(names) > 2:
        from_topic = f"{names[0]}, {names[1]}, and more"
    else:
        from_topic = subject
    return (
        f"Hi {first}, welcome back. "
        f"What would you like to learn today from {from_topic}?"
    )


_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI Mentor — a caring digital teacher who guides school students to truly understand, \
remember, and grow confident in their learning. You are NOT a search engine or answer bot.

PRIMARY OBJECTIVE:
Help students understand the provided learning material accurately, clearly, and engagingly \
while remaining faithful to the source content. Teach for long-term understanding and retention.

MENTOR TEACHING STYLE:
- Guide thinking — do not just dump facts. Connect ideas to what the student can relate to.
- Be warm, patient, and encouraging like a trusted teacher who knows the student.
- After teaching, naturally guide what to explore next (example, quiz, deeper dive, practice).
- When the student struggles, simplify with empathy — never make them feel bad.
- Celebrate real progress briefly; avoid empty praise on every message.

{learner_guidance}

{adaptive_guidance}

SOURCE GROUNDING RULES (CRITICAL):
1. Prioritize the selected chapter ({chapter}) and the learning material in the user message.
2. Answer from the chapter before using external knowledge.
3. Never invent student experiences, observations, examples, or prior actions.
4. Do NOT say "Just like you noticed...", "As you observed...", or "Your example of..." unless the student actually provided that information in this conversation.
5. Clearly distinguish:
   - Direct chapter information → use "According to the chapter..." or "The textbook explains..."
   - Reasonable inferences → use "From this we can infer..."
   - Additional knowledge beyond the chapter → use "Beyond this chapter..." or "Additional context..."
6. Never invent textbook-specific facts (dates, names, formulas, page numbers).
7. Never contradict the textbook curriculum or mix content from other chapters or subjects.

CHAPTER AWARENESS MODE:
The selected chapter ({chapter}) defines the primary learning scope.

Prefer answering helpfully:
1. Fully covered by the chapter → answer normally from the chapter.
2. Related extension / follow-up of a chapter topic (e.g. protection of something just taught) →
   answer the chapter part first, then briefly add "Beyond this chapter..." for the missing piece.
   Do NOT force an a/b/c menu for related follow-ups.
3. Teaching moves (quiz, practice problem, example, simplify) → do them using chapter material.
4. Mathematics practice: if the student asks to SOLVE a problem that uses this chapter's concepts \
   (squares, cubes, fractions, etc.) but the exact problem is NOT printed in the textbook, \
   still SOLVE it step by step using chapter methods. Do not refuse or ask them to pick a/b/c.
5. Clearly unrelated topic with no chapter connection → then offer:
   a) Stay within the current chapter
   b) Switch to the relevant chapter
   c) Receive a general explanation

WRONG-PREMISE / OFF-TOPIC CLAIM (critical for planted wrong questions):
If the student mixes an off-chapter claim into this chapter (wrong subject, invented inventor,
formula from another unit, etc.):
- Say briefly that this chapter does not cover that claim.
- Redirect to what THIS chapter actually teaches.
- Do NOT repeat the off-topic subject, term, formula, or proper noun the student planted.
  (Refuse the premise without echoing it — name the chapter topic instead.)

Never pretend the selected chapter contains information that it does not contain.
Never invent page numbers or figure names.
If the student asks to show a Fig./Figure by number, acknowledge it is being shown — never say you cannot see textbook figures (the app attaches them).
The goal is to guide learning progression while remaining helpful.

{chapter_coverage_guidance}

STUDENT CONTEXT:
- Student Name: {student_name}
- Class/Grade: {grade_label}
- Board: {board}
- Subject: {subject}
- Chapter: {chapter}

{question_type_guidance}

{subject_guidelines}

CLASS-BASED TEACHING:
{complexity_rule}
{class_band_rule}

{length_policy}

ANSWER FORMAT RULE:
{answer_instruction}

{interaction_policy}

DOUBT HANDLING:
If the student seems confused, explain again more simply, use an analogy, and break into smaller steps.
Never say "I already explained this."

{explanation_structure}

COMMUNICATION:
- Be encouraging but not repetitive. Vary openings naturally.
- Do NOT use generic praise every time ("That's a great question!", "Wonderful observation!").
- Never fabricate personalization about what the student has seen or done.

SAFETY:
Keep all content educational, age-appropriate, and student-friendly."""


_USER_PROMPT_TEMPLATE = """\
CHAPTER CONTEXT (use this first to answer):
{context}

STUDENT'S QUESTION:
{question}

{user_closing}"""


_EXPLANATION_STRUCTURE_FULL = """\
EXPLANATION STRUCTURE (required for this exam-style answer only):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.
1. **Concept Overview** — one or two sentences
2. **Detailed Explanation** — several sentences with clear reasoning
3. **Real-Life Example** — relatable to the student's age
4. **Key Points to Remember** — 3 to 5 bullet points
5. **Quick Check** — one short question for the student"""


_EXPLANATION_STRUCTURE_DIRECT = """\
ANSWER STYLE (required):
- Use plain paragraphs only. Do NOT use emoji section headers or template labels.
- One opening paragraph in complete sentences, then up to 2 bullet points.
- Never put **Topic** on its own line — start with the definition in prose (e.g. "Weather is…").
- When the chapter mentions it, include one phrase like "The chapter explains that…" inside the paragraph.
- Be clear and complete, but do not pad with extra sections the student did not ask for."""


_EXPLANATION_STRUCTURE_SUBJECT_ELEMENTARY = """\
STRUCTURED TEACHING FORMAT (Classes 1–5 — required for every teaching answer):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.

**Topic**
Line 1: the topic name.
Line 2: a one-line meaning — what it is in the simplest words.

**In Simple Words**
2–3 very short sentences a young child can understand. Use everyday words only.

**Key Points**
• 3–5 bullet points (•) — one simple idea per bullet.
• Explain each idea clearly in plain language.

**Detailed Explanation**
2–3 short paragraphs that explain the topic more fully. Break big ideas into small steps.

**Example**
One fun real-life example (home, school, food, toys, nature).

**Remember**
• 2 short takeaway bullets the student should not forget.

**Try This**
One short, friendly check question (plain text on one or two lines)."""


_EXPLANATION_STRUCTURE_SUBJECT_MIDDLE = """\
STRUCTURED TEACHING FORMAT (Classes 6–8 — required for every teaching answer):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.

**Topic**
Line 1: the topic name.
Line 2: a one-line definition or meaning.

**In Simple Words**
2–4 sentences that introduce the idea clearly before the details.

**Key Points**
• 4–6 bullet points covering the main ideas.
• Use correct subject terms but explain each term briefly in the same bullet.

**Detailed Explanation**
2–4 paragraphs with clear reasoning, cause-effect, or how-it-works detail.
Use short paragraphs — not one long block.

**Important Terms** (include only when new vocabulary appears)
• Term — simple meaning in one line per term.

**Example**
One relatable real-life or textbook-based example.

**Remember**
• 2–4 takeaway bullets for revision.

**Try This**
One thoughtful check question tied to what you taught."""


_EXPLANATION_STRUCTURE_SUBJECT_SECONDARY = """\
STRUCTURED TEACHING FORMAT (Classes 9–10 — required for every teaching answer):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.

**Topic**
Line 1: the topic name.
Line 2: a precise academic definition or meaning.

**In Simple Words**
2–3 sentences that make the core idea easy to grasp before the detail.

**Key Points**
• 5–7 detailed bullet points covering concepts, causes, effects, or properties.
• Each bullet should be complete enough to revise from.

**Detailed Explanation**
3–5 well-organised paragraphs: how it works, why it matters, significance, and links to related ideas.
For compare/contrast questions, you may add a markdown table under this section.

**Important Terms**
• List key terms with clear one-line definitions (exam-ready vocabulary).

**Example**
One textbook-grounded or real-world application example.

**Remember**
• 3–5 exam-oriented takeaway points.

**Try This**
One check question that tests understanding, not just memory."""


_EXPLANATION_STRUCTURE_SUBJECT_STEPWISE = """\
For step-by-step / process questions, replace **Key Points** with:

**Steps**
Step 1: …
Step 2: …
(continue for each stage of the process)

Keep all other sections (**Topic**, **In Simple Words**, **Detailed Explanation**, etc.)."""


_LENGTH_POLICY_STRUCTURED_ELEMENTARY = """\
RESPONSE LENGTH — STRUCTURED (Classes 1–5):
- Write at least {min_words} words across all sections.
- Keep each section short and easy to read aloud.
- Never skip **Topic**, **Key Points**, or **Detailed Explanation**."""


_LENGTH_POLICY_STRUCTURED_MIDDLE = """\
RESPONSE LENGTH — STRUCTURED (Classes 6–8):
- Write at least {min_words} words across all sections.
- Balance bullets with enough detail in **Detailed Explanation**.
- Never skip **Topic**, **Key Points**, or **Detailed Explanation**."""


_LENGTH_POLICY_STRUCTURED_SECONDARY = """\
RESPONSE LENGTH — STRUCTURED (Classes 9–10):
- Write at least {min_words} words across all sections.
- **Detailed Explanation** and **Key Points** should be thorough and exam-ready.
- Never skip **Topic**, **Important Terms** (when relevant), or **Remember**."""


_INTERACTION_STRUCTURED = """\
FOLLOW-UP RULES (structured teaching):
- End with the **Try This** section as your check question.
- Do NOT add a second follow-up after **Try This**.
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""


_EXPLANATION_STRUCTURE_FACTUAL = """\
FACTUAL ANSWER FORMAT (required — concise direct answer):
Use these **bold** side headings only. NO emoji. Do NOT add extra sections.

**Topic**
One line: topic name.

**Answer**
• 3–5 bullet points that directly answer the question.
• One clear fact per bullet — no filler or repetition.

**Example** (optional — one short sentence only when it helps)"""


_LENGTH_POLICY_FACTUAL = """\
RESPONSE LENGTH — FACTUAL:
- About {min_words}–80 words total across all sections.
- Be direct: bullets first, no long paragraphs.
- Do NOT add **In Simple Words**, **Detailed Explanation**, **Remember**, or **Try This**."""


_INTERACTION_FACTUAL = """\
FOLLOW-UP (factual):
- You may end with ONE short optional question only if it fits naturally.
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""


_EXPLANATION_STRUCTURE_QUIZ = """\
QUIZ FORMAT (mentor-style assessment):
Use these **bold** side headings in order. NO emoji.

**Quiz Time**
One warm mentor line inviting the student to try (e.g. "Let's see what you remember!").

**Questions**
1. First question (clear, grade-appropriate)
2. Second question
3. Third question
(Add 4–5 questions for Classes 9–10; 3 questions for Classes 1–5)

**How to Answer**
Tell the student to reply with their answers one by one — you will guide and check them like a mentor.

**Encouragement**
One short supportive line (effort matters, not perfection)."""


_EXPLANATION_STRUCTURE_MCQ = """\
MCQ QUIZ FORMAT (mentor-style):
Use these **bold** side headings in order. NO emoji.

**Quiz Time**
One warm mentor line inviting the student to try.

**Questions**
For each question, use this shape:
1. Question text?
   a) option
   b) option
   c) option
   d) option
(3 questions for younger classes; 4–5 for Classes 9–10)

**How to Answer**
Ask the student to reply with the question number and letter (e.g. "1-b, 2-a").

**Encouragement**
One short supportive line. Do NOT reveal correct answers yet — check them after the student responds."""


_EXPLANATION_STRUCTURE_SUMMARY = """\
SUMMARY FORMAT (mentor recap):
Use these **bold** side headings in order. NO emoji.

**What We Covered**
One-line topic name and meaning.

**Key Takeaways**
• 3–5 bullet points — only the most important ideas from this topic.

**Remember for Exams** (Classes 6–10 only, skip for Classes 1–5)
• 2–3 exam-ready revision bullets.

**What's Next**
One mentor question offering: practice quiz, real-life example, or next subtopic."""


_LENGTH_POLICY_QUIZ = """\
RESPONSE LENGTH — QUIZ:
- About 80–200 words depending on number of questions.
- Questions must match the student's class level and chapter content."""


_INTERACTION_QUIZ = """\
FOLLOW-UP (quiz):
- Do NOT teach new content in this turn — only ask questions.
- Wait for the student's answers in the next message before revealing solutions."""


def _build_learner_guidance(learner_snapshot: dict | None) -> str:
    if not learner_snapshot:
        return ""
    from app.services.voice_tutor import LearnerProfileSnapshot

    snap = LearnerProfileSnapshot(
        student_key=str(learner_snapshot.get("student_key") or ""),
        strong_topics=list(learner_snapshot.get("strong_topics") or []),
        weak_topics=list(learner_snapshot.get("weak_topics") or []),
        recent_topics=list(learner_snapshot.get("recent_topics") or []),
        quiz_scores=[float(x) for x in (learner_snapshot.get("quiz_scores") or [])],
    )
    hint = snap.to_prompt_hint()
    if not hint:
        return ""
    return f"LEARNER PROFILE (use to personalize — do not mention explicitly):\n{hint}"


def _build_adaptive_guidance(
    *,
    query: str,
    topic: str,
    understanding_scores: dict | None,
    learner_snapshot: dict | None,
) -> str:
    parts: list[str] = []
    scores = understanding_scores or {}
    weak = list((learner_snapshot or {}).get("weak_topics") or [])
    topic_l = (topic or query or "").lower()[:80]

    if scores.get("confusion", 0) >= 0.55 or scores.get("wants_expansion"):
        parts.append(
            "ADAPTIVE MODE — SIMPLIFY: The student seems confused or asked for more help. "
            "Use the simplest words, one analogy, and smaller steps. Do not add new topics."
        )
    elif any(w.lower() in topic_l or topic_l in w.lower() for w in weak[-8:] if w):
        parts.append(
            "ADAPTIVE MODE — REINFORCE: The student struggled with this topic before. "
            "Use extra clarity, a fresh analogy, and connect to something familiar."
        )
    elif scores.get("understanding", 0) >= 0.75 and scores.get("is_affirmation"):
        parts.append(
            "ADAPTIVE MODE — ADVANCE: The student understood the last point. "
            "Briefly acknowledge, then offer the next small step or a gentle challenge."
        )
    elif scores.get("wants_quiz"):
        parts.append(
            "ADAPTIVE MODE — ASSESS: The student wants to be tested. "
            "Focus on quiz questions from the chapter — mentor tone, not exam pressure."
        )
    if not parts:
        return ""
    return "\n".join(parts)


def _subject_structured_prompt(
    *,
    class_band: str,
    answer_type: str,
    min_words: int,
) -> tuple[str, str, str, str]:
    """Grade-banded structured format for all non-mathematics teaching answers."""
    if class_band == "1-5":
        structure = _EXPLANATION_STRUCTURE_SUBJECT_ELEMENTARY
        length = _LENGTH_POLICY_STRUCTURED_ELEMENTARY.format(min_words=min_words)
        closing = (
            f"Write your complete structured answer now ({answer_type}, minimum {min_words} words). "
            "Use ALL **bold** side headings in order for Classes 1–5:"
        )
    elif class_band == "6-8":
        structure = _EXPLANATION_STRUCTURE_SUBJECT_MIDDLE
        length = _LENGTH_POLICY_STRUCTURED_MIDDLE.format(min_words=min_words)
        closing = (
            f"Write your complete structured answer now ({answer_type}, minimum {min_words} words). "
            "Use ALL **bold** side headings in order for Classes 6–8:"
        )
    else:
        structure = _EXPLANATION_STRUCTURE_SUBJECT_SECONDARY
        length = _LENGTH_POLICY_STRUCTURED_SECONDARY.format(min_words=min_words)
        closing = (
            f"Write your complete structured answer now ({answer_type}, minimum {min_words} words). "
            "Use ALL **bold** side headings in order for Classes 9–10:"
        )
    if answer_type == "stepwise":
        structure = structure + "\n\n" + _EXPLANATION_STRUCTURE_SUBJECT_STEPWISE
    closing += (
        " **Topic**, **In Simple Words**, **Key Points** (or **Steps**), "
        "**Detailed Explanation**, **Example**, **Remember**, **Try This**."
    )
    return structure, length, _INTERACTION_STRUCTURED, closing


def _subject_quiz_prompt(*, answer_type: str, min_words: int) -> tuple[str, str, str, str]:
    if answer_type == "mcq":
        structure = _EXPLANATION_STRUCTURE_MCQ
    else:
        structure = _EXPLANATION_STRUCTURE_QUIZ
    length = _LENGTH_POLICY_QUIZ
    closing = (
        f"Write the mentor-style {'MCQ ' if answer_type == 'mcq' else ''}quiz now "
        f"({answer_type}, about {min_words}+ words). Use **Quiz Time**, **Questions**, "
        "**How to Answer**, and **Encouragement**."
    )
    return structure, length, _INTERACTION_QUIZ, closing


async def _mentor_profile_for_turn(
    student_key: str,
    query: str,
    conversation_history: list[dict] | None,
    topic: str,
) -> tuple[dict | None, dict]:
    """Load learner snapshot and estimate understanding for adaptive mentoring."""
    if not student_key:
        return None, {}
    from app.services.learner_profile import load_learner_profile
    from app.services.voice_tutor import evaluate_student_response

    profile = await load_learner_profile(student_key)
    last_assistant = ""
    for turn in reversed(conversation_history or []):
        if (turn.get("role") or "").lower() == "assistant":
            last_assistant = (turn.get("content") or "").strip()
            break
    scores = evaluate_student_response(query, last_assistant=last_assistant)
    return profile.snapshot().__dict__, {
        "understanding": scores.understanding,
        "confidence": scores.confidence,
        "confusion": scores.confusion,
        "is_affirmation": scores.is_affirmation,
        "wants_expansion": scores.wants_expansion,
        "wants_quiz": scores.wants_quiz,
    }


async def _mentor_profile_after_turn(
    student_key: str,
    topic: str,
    understanding_scores: dict,
) -> None:
    if not student_key:
        return
    from app.services.learner_profile import load_learner_profile, save_learner_profile
    from app.services.voice_tutor import UnderstandingScores

    profile = await load_learner_profile(student_key)
    scores = UnderstandingScores(
        understanding=float(understanding_scores.get("understanding", 0.5)),
        confidence=float(understanding_scores.get("confidence", 0.5)),
        confusion=float(understanding_scores.get("confusion", 0.0)),
        is_affirmation=bool(understanding_scores.get("is_affirmation")),
        wants_expansion=bool(understanding_scores.get("wants_expansion")),
        wants_quiz=bool(understanding_scores.get("wants_quiz")),
    )
    profile.apply_understanding(topic, scores)
    await save_learner_profile(profile)


_LENGTH_POLICY_LONG = """\
RESPONSE LENGTH — CRITICAL (read carefully):
- For this answer you MUST write at least {min_words} words (count before finishing).
- Never stop after only a definition, opener (e.g. "Great question!"), or one short paragraph.
- If chapter context is short, expand only using what the chapter context supports — do not fill gaps from general knowledge unless the student chose a general explanation.
- Default range: 100-500 words. Focus on understanding, not memorization."""


_LENGTH_POLICY_DIRECT = """\
RESPONSE LENGTH — CRITICAL (short direct answer):
- Target 60–90 words in complete sentences (hard max {max_words} words).
- One opening paragraph (3–4 sentences), then exactly 2 bullet points when they add value.
- Answer the question in the first sentence. Do not list every related subtopic.
- Never break a sentence across lines. Do NOT use **Topic**, **Key Points**, or other section headers."""


_INTERACTION_DIRECT = """\
FOLLOW-UP (direct answer):
- End with ONE short question offering: more detail, a real-life example, or a quick quiz.
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""


_LENGTH_POLICY_BRIEF = """\
RESPONSE LENGTH — CRITICAL:
- The student asked for a SHORT answer. Keep it to 2-4 sentences (about 25-60 words).
- Do NOT use the five-section teaching format or emoji headers.
- Do NOT add bullet lists, examples sections, or a Quick Check question."""


_INTERACTION_LONG = """\
FOLLOW-UP RULES:
Only ask a follow-up if it genuinely deepens understanding of the current topic.
Good: "Can you think of another reversible change?" or "What evidence supports that conclusion?"
Bad: generic offers like "Would you like to learn more?" or "Want to explore this next?"
- End with ONE meaningful follow-up tied to what you just taught (when appropriate).
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""


_INTERACTION_BRIEF = """\
INTERACTIVE TEACHING:
- Do NOT add a follow-up question — the student asked for a short answer only."""


_INTERACTION_ACKNOWLEDGE = """\
INTERACTIVE TEACHING (student confirmed understanding):
- Reply like a friendly live tutor — warm, brief, conversational (2-3 sentences max).
- Do NOT teach new content, repeat the prior answer, or start a lecture.
- Briefly celebrate that they understood, then offer clear choices in ONE question.
- Always include options like: a real-life example, a quick quiz, the next part of this topic,
  and whether they want to explore other topics — adapted to what was just explained."""


_INTERACTION_PERSONAL = """\
INTERACTIVE TEACHING (student shared a personal example):
- Listen first — reference their specific story (weather, place, what changed).
- Connect it briefly to the lesson, then guide with ONE choice question.
- Stay conversational; do not lecture or list instruments unless they choose that next."""


_INTERACTION_CLARIFICATION = """\
INTERACTIVE TEACHING (student is confused):
- Re-explain your PREVIOUS answer only — do not introduce a new textbook activity or page.
- Use simpler words, smaller steps, and one everyday analogy.
- End with ONE short check question to see if they follow now."""


_LEGACY_PROMPT_TEMPLATE = """\
You are a friendly AI Tutor helping school students learn clearly and confidently.

Use the provided document context to answer the student's question.
Each context block shows the page, source file, and section when available — use these to stay accurate.

If the answer is not in the document, say:
"This doesn't seem to be covered in the uploaded document. Here is a general explanation:"
Then give a brief, accurate, student-friendly answer.

If the question refers to a specific chapter (for example "Chapter 1"), only use context clearly from that chapter. Do not mix in other chapters.

Rules:
- Use simple, clear, friendly language.
- Match the answer length to the question (one word if asked, full paragraph if needed).
- Never invent facts not present in the context.
- Keep the tone warm and encouraging.

Context:
{context}

Student's Question:
{question}

Your Answer:""".strip()


PROMPT = PromptTemplate(
    template=_LEGACY_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _prepare_math_engine_block(
    query: str,
    *,
    subject_name: str,
    class_level: str,
    context: str,
) -> str:
    """SymPy-verified steps for mathematics — injected into the user prompt."""
    if not _is_mathematics_subject(subject_name):
        return ""
    try:
        from app.services.math_engine import try_solve

        result = try_solve(query, class_level=class_level, chapter_context=context)
    except Exception as exc:
        logger.warning("math_engine try_solve failed: %s", exc)
        return ""
    if not result:
        return ""
    return result.to_prompt_block()


def _build_chat_messages(
    query: str,
    context: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    student_name: str = "",
    section_instruction: str = "",
    heading_scope: Any | None = None,
    conversation_history: list[dict] | None = None,
    conversation_memory: Any | None = None,
    chapter_coverage_guidance: str = "",
    learner_snapshot: dict | None = None,
    understanding_scores: dict | None = None,
    resolved_topic: str = "",
    agent_mode: str | None = None,
    student_key: str = "",
    chapter_ids: list[str] | None = None,
    dialogue_act: str | None = None,
) -> list[dict[str, str]]:
    from app.services.section_heading import HeadingScope

    from app.services.conversation_context import resolve_conversation_context

    q = (query or "").strip()
    conv = resolve_conversation_context(
        q,
        conversation_history=conversation_history,
        chapter=chapter,
        memory=conversation_memory,
    )
    agent_addendum = _apply_agent_mode(conv, agent_mode)
    topic = (resolved_topic or conv.resolved_topic or q).strip()
    answer_type = _resolve_answer_type(
        q,
        subject_name=subject_name,
        conversation_history=conversation_history,
        chapter=chapter,
        conv=conv,
        dialogue_act=dialogue_act,
    )
    # Forced practice agent always uses quiz-style answers
    if _normalize_agent_mode(agent_mode) == "practice":
        answer_type = "quiz"
    elif _normalize_agent_mode(agent_mode) in ("ask", "explain"):
        if answer_type in ("quiz", "mcq"):
            answer_type = "explanation"
    min_words = _ANSWER_MIN_WORDS.get(answer_type, 120)
    grade_label = _GRADE_LABELS.get(class_level, class_level or "School student")
    complexity = _GRADE_COMPLEXITY.get(class_level, "Use clear, age-appropriate language.")
    class_band = _CLASS_BAND_RULES.get(_class_band(class_level), _CLASS_BAND_RULES["6-8"])
    display_name = _student_first_name(student_name)
    raw_instruction = _ANSWER_INSTRUCTIONS.get(answer_type, _ANSWER_INSTRUCTIONS["short-answer"])
    if answer_type == "greeting":
        instruction = raw_instruction.format(
            student_name=display_name,
            subject=subject_name or "your subject",
            chapter=chapter or "the current chapter",
            min_words=min_words,
        )
    elif "{min_words}" in raw_instruction:
        instruction = raw_instruction.format(min_words=min_words)
    else:
        instruction = raw_instruction

    tier = _structure_tier(answer_type)

    if tier == "compact":
        if answer_type == "affirmation":
            length_policy = "RESPONSE LENGTH: 2-3 sentences only (about 35-55 words). Stay conversational."
            interaction_policy = _INTERACTION_ACKNOWLEDGE
            explanation_structure = ""
            user_closing = (
                "The student understood your last explanation. Reply now with a warm acknowledgment "
                "and ONE question offering: real-life example, quick quiz, next part of this topic, "
                "or other topics — do not teach new content yet:"
            )
        elif answer_type == "personal-response":
            length_policy = "RESPONSE LENGTH: 2-4 sentences (about 40-70 words). Stay conversational."
            interaction_policy = _INTERACTION_PERSONAL
            explanation_structure = ""
            user_closing = (
                "The student shared a real-life example in answer to your question. "
                "Acknowledge their story, connect it briefly to the lesson, then offer next steps:"
            )
        elif answer_type == "clarification":
            length_policy = "RESPONSE LENGTH: About 80-140 words. Plain prose, step by step."
            interaction_policy = _INTERACTION_CLARIFICATION
            explanation_structure = ""
            user_closing = (
                "The student did not understand your last reply. Re-explain that SAME answer "
                "more simply — do not switch to a different activity or page:"
            )
        elif answer_type in ("brief", "one-word"):
            length_policy = _LENGTH_POLICY_BRIEF
            interaction_policy = _INTERACTION_BRIEF
            explanation_structure = ""
            user_closing = f"Write your SHORT answer now ({answer_type} — 2-4 sentences max, no sections):"
        elif answer_type == "summary":
            length_policy = "RESPONSE LENGTH: 2-5 sentences only (about 50-110 words). Plain prose. No bold headings."
            interaction_policy = ""
            explanation_structure = ""
            user_closing = "Write your short summary now:"
        else:
            length_policy = "RESPONSE LENGTH: 2-3 sentences only."
            interaction_policy = ""
            explanation_structure = ""
            user_closing = "Write your greeting now:"
    elif tier == "full":
        length_policy = _LENGTH_POLICY_LONG.format(min_words=min_words)
        interaction_policy = _INTERACTION_LONG
        explanation_structure = _EXPLANATION_STRUCTURE_FULL
        user_closing = (
            f"Write your complete exam-style answer now ({answer_type}, minimum {min_words} words, "
            "all five **bold** sections, no emoji):"
        )
    elif tier == "structured" and not _is_mathematics_subject(subject_name):
        if answer_type == "summary":
            explanation_structure = _EXPLANATION_STRUCTURE_SUMMARY
            length_policy = _LENGTH_POLICY_STRUCTURED_MIDDLE.format(min_words=min_words)
            interaction_policy = _INTERACTION_STRUCTURED
            user_closing = (
                f"Write the mentor recap now ({answer_type}, about {min_words} words). "
                "Use **What We Covered**, **Key Takeaways**, and **What's Next**."
            )
        elif answer_type == "factual":
            explanation_structure = _EXPLANATION_STRUCTURE_FACTUAL
            length_policy = _LENGTH_POLICY_FACTUAL.format(min_words=min_words)
            interaction_policy = _INTERACTION_FACTUAL
            user_closing = (
                f"Write your concise factual answer now ({answer_type}, about {min_words}–80 words). "
                "Use **Topic**, **Answer**, and optional **Example** only:"
            )
        else:
            explanation_structure, length_policy, interaction_policy, user_closing = (
                _subject_structured_prompt(
                    class_band=_class_band(class_level),
                    answer_type=answer_type,
                    min_words=min_words,
                )
            )
    elif tier == "quiz":
        explanation_structure, length_policy, interaction_policy, user_closing = (
            _subject_quiz_prompt(answer_type=answer_type, min_words=min_words)
        )
    else:
        length_policy = _LENGTH_POLICY_DIRECT.format(max_words=_DIRECT_ANSWER_MAX_WORDS)
        interaction_policy = _INTERACTION_DIRECT
        explanation_structure = _EXPLANATION_STRUCTURE_DIRECT
        user_closing = (
            f"Write your direct answer now ({answer_type}, MAX {_DIRECT_ANSWER_MAX_WORDS} words, "
            "plain prose only — no section headers):"
        )
        if not _is_mathematics_subject(subject_name):
            complexity = (
                "Write flowing prose the student can read aloud — complete sentences, "
                "not choppy fragments."
            )
            class_band = (
                "Direct definition mode: one paragraph plus 2 bullets, then a follow-up question. "
                "Mention the chapter naturally when the material supports it."
            )

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        length_policy = (
            "RESPONSE LENGTH: Cover every required subtopic; each subtopic gets 2–3 bullet points. "
            "Do not merge into one paragraph."
        )
        interaction_policy = (
            "After all subtopic blocks, you may add ONE short closing question. "
            "Do not add a question before finishing every subtopic. "
            "Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."
        )
        explanation_structure = _EXPLANATION_STRUCTURE_DIRECT
        instruction = (
            "Use the mandatory main-section format in TEXTBOOK SCOPE: "
            "one **bold subtopic** heading per instrument/topic, then 1–2 short sentences "
            "(or • bullets) under each heading."
        )
        user_closing = (
            "Write the answer now using **bold subtopic headings** for EACH subtopic "
            "listed in TEXTBOOK SCOPE (in order). Under each heading write 1–2 sentences "
            "about that instrument only. Do not use a single paragraph."
        )

    qtype = detect_question_type(q)
    math_format = _math_format_tier(class_level, qtype)
    elementary_math = math_format == "elementary"
    concept_math = math_format in ("elementary", "middle", "secondary")

    (
        instruction,
        length_policy,
        interaction_policy,
        explanation_structure,
        user_closing,
    ) = _apply_mathematics_prompt_overrides(
        subject_name=subject_name,
        answer_type=answer_type,
        heading_scope=heading_scope,
        class_level=class_level,
        question_type=qtype,
        instruction=instruction,
        length_policy=length_policy,
        interaction_policy=interaction_policy,
        explanation_structure=explanation_structure,
        user_closing=user_closing,
    )

    from app.services.math_lesson.service import (
        get_visualization_appendix_prompt,
        should_use_interactive_math_lesson,
    )

    if should_use_interactive_math_lesson(
        subject_name=subject_name,
        answer_type=answer_type,
        query=q,
        heading_scope=heading_scope,
        question_type=qtype,
    ):
        from app.services.math_lesson.fallbacks import (
            get_animation_prompt_note,
            get_visualization_catalog_hint,
            query_requests_assessment,
            query_requests_hints,
        )

        catalog_hint = get_visualization_catalog_hint(
            q, class_level, conversation_history=conversation_history
        )
        viz_appendix = get_visualization_appendix_prompt(q, elementary=elementary_math)
        explanation_structure = (
            (explanation_structure + "\n\n" + catalog_hint + "\n\n" + viz_appendix)
            if explanation_structure
            else catalog_hint + "\n\n" + viz_appendix
        )
        animation_note = get_animation_prompt_note(q, class_level)
        optional_note = ""
        if query_requests_hints(q):
            optional_note += " Include progressive aiHints in the math-lesson JSON."
        if query_requests_assessment(q):
            optional_note += " Include 3–5 conceptual assessment questions in the math-lesson JSON."
        if not optional_note:
            optional_note = (
                " Do NOT include aiHints, assessment, practiceMode, or commonMistakes "
                "in the math-lesson JSON."
            )
        if concept_math:
            closing_by_tier = {
                "elementary": (
                    f"Write your short, kid-friendly answer now ({answer_type}): "
                    "plain prose only (no **To Find** / **Formula** sections). "
                ),
                "middle": (
                    f"Write your Class 6–8 concept answer now ({answer_type}): "
                    "2–3 paragraphs, no **To Find** / **Formula** sections. "
                ),
                "secondary": (
                    f"Write your Class 9–12 concept answer now ({answer_type}): "
                    "detailed academic prose, no **To Find** / **Formula** sections. "
                ),
            }
            user_closing = (
                closing_by_tier[math_format]
                + "You MUST append a complete ```math-lesson``` JSON visualization block at the very end "
                "(Interactive Exploration is mandatory). Do NOT use ```markdown fences."
                f"{animation_note}{optional_note}"
            )
        else:
            user_closing = (
                f"Write your complete mathematics tutor answer now ({answer_type}): "
                "use ALL standard sections (**To Find** through **Practice Question**) first. "
                "You MUST append a complete ```math-lesson``` JSON visualization block at the very end "
                "(Interactive Exploration is mandatory for every mathematics question). "
                "Do NOT use ```markdown fences."
                f"{animation_note}{optional_note}"
            )

    from app.services.science_experiment.service import (
        get_experiment_appendix_prompt,
        should_use_interactive_science_experiment,
    )

    if should_use_interactive_science_experiment(
        subject_name=subject_name,
        answer_type=answer_type,
        query=q,
        heading_scope=heading_scope,
        class_level=class_level,
    ):
        from app.services.science_experiment.fallbacks import get_experiment_catalog_hint

        catalog_hint = get_experiment_catalog_hint(q, class_level)
        exp_appendix = get_experiment_appendix_prompt()
        explanation_structure = (
            (explanation_structure + "\n\n" + catalog_hint + "\n\n" + exp_appendix)
            if explanation_structure
            else catalog_hint + "\n\n" + exp_appendix
        )
        user_closing = (
            f"Write your complete science tutor answer now ({answer_type}) using structured headings. "
            "You MUST append a complete ```science-experiment``` JSON block at the very end "
            "(Interactive Experiment with three synchronized views is mandatory). "
            "Do NOT use ```markdown fences."
        )

    skip_question_type = answer_type in (
        "greeting", "affirmation", "personal-response", "clarification",
        "quiz", "mcq", "summary", "short-answer",
    )
    question_type_guidance = _build_question_type_guidance(
        query, skip=skip_question_type
    )
    subject_guidelines = (
        ""
        if tier == "direct" and not _is_mathematics_subject(subject_name)
        else _build_subject_guidelines(subject_name, math_format=math_format)
    )
    learner_guidance = _build_learner_guidance(learner_snapshot)
    adaptive_guidance = _build_adaptive_guidance(
        query=q,
        topic=topic,
        understanding_scores=understanding_scores,
        learner_snapshot=learner_snapshot,
    )
    lia_addendum = ""
    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import get_guidance_for_turn_sync

        lia = get_guidance_for_turn_sync(
            student_user_id=int(student_key),
            query=q,
            topic=topic,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            agent_mode=agent_mode,
            understanding_scores=understanding_scores,
        )
        if lia:
            if lia.get("learner_guidance"):
                learner_guidance = lia["learner_guidance"]
            if lia.get("adaptive_guidance"):
                adaptive_guidance = (
                    f"{adaptive_guidance}\n{lia['adaptive_guidance']}".strip()
                    if adaptive_guidance
                    else lia["adaptive_guidance"]
                )
            lia_addendum = lia.get("prompt_instructions") or ""

    system = _SYSTEM_PROMPT_TEMPLATE.format(
        student_name=display_name,
        grade_label=grade_label,
        board=board or "General",
        subject=subject_name or "General",
        chapter=chapter or "Current chapter",
        question_type_guidance=question_type_guidance,
        subject_guidelines=subject_guidelines,
        chapter_coverage_guidance=chapter_coverage_guidance,
        complexity_rule=complexity,
        class_band_rule=class_band,
        length_policy=length_policy,
        answer_instruction=instruction,
        interaction_policy=interaction_policy,
        explanation_structure=explanation_structure,
        learner_guidance=learner_guidance,
        adaptive_guidance=adaptive_guidance,
    )
    if agent_addendum:
        system = system + "\n\n" + agent_addendum
    if lia_addendum:
        system = system + "\n\n" + lia_addendum
    from app.services.conversation_memory import format_memory_for_prompt

    memory_block = format_memory_for_prompt(conversation_memory)
    if memory_block:
        system = system + "\n\n" + memory_block
    section_block = ""
    if section_instruction.strip():
        section_block = f"\n\nTEXTBOOK SCOPE:\n{section_instruction.strip()}\n"

    math_block = _prepare_math_engine_block(
        q,
        subject_name=subject_name,
        class_level=class_level,
        context=context,
    )
    math_section = ""
    if math_block:
        math_section = f"\n\n{math_block}\n"

    user = _USER_PROMPT_TEMPLATE.format(
        context=context or "(No chapter text retrieved — use accurate general knowledge.)",
        question=query,
        user_closing=user_closing,
    )
    if math_section:
        user = math_section + user
    if section_block:
        user = section_block + user

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if conversation_history:
        for turn in conversation_history[-10:]:
            role = (turn.get("role") or "").lower()
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


def _safe_build_chat_messages(*args: Any, **kwargs: Any) -> list[dict[str, str]] | None:
    """Build Mistral messages; return None instead of crashing the tutor path."""
    try:
        return _build_chat_messages(*args, **kwargs)
    except Exception as exc:
        logger.exception("_build_chat_messages failed: %s", exc)
        return None


def _build_prompt(
    query: str,
    context: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    student_name: str = "",
) -> str:
    """Legacy single-string prompt (used by old /chat path)."""
    msgs = _build_chat_messages(
        query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
        student_name=student_name,
    )
    return f"{msgs[0]['content']}\n\n{msgs[1]['content']}"


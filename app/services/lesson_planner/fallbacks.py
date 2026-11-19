from __future__ import annotations

import re
from typing import Any

from app.modules.teacher.lesson_planner.constants import ArtifactType
from app.services.lesson_planner.state import PlannerState


def _fallback_lesson_plan_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    duration = int(state.get("duration_minutes") or 45)
    objectives_raw = state.get("learning_objectives") or ""
    objectives = [o.strip() for o in objectives_raw.split("\n") if o.strip()] or [
        f"Understand key ideas in {chapter}"
    ]
    engage = max(8, duration // 6)
    explore = max(12, duration // 4)
    explain = max(15, duration // 3)
    elaborate = max(10, duration // 5)
    evaluate = max(5, duration - engage - explore - explain - elaborate)

    lines = [
        f"# Lesson Plan: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        f"Duration: {duration} minutes",
        "",
        "## Learning Objectives",
        *[f"- {obj}" for obj in objectives],
        "",
        "## Prerequisite Knowledge",
        f"- Prior knowledge from earlier {subject} topics",
        "",
        "## Materials Required",
        "- Textbook",
        "- Whiteboard / projector",
        "- Student notebooks",
        "",
        "## Key Vocabulary",
        f"- {chapter}",
        "",
        "## Lesson Flow",
        "",
        "### Engage",
        f"- **Duration:** {engage} minutes",
        f"- **Teacher Activity:** Ask what students already know about {chapter}.",
        f"- **Student Activity:** Share prior knowledge in pairs.",
        "- **Guiding Questions:** What do you notice about this topic in daily life?",
        "- **Expected Student Responses:** Everyday examples and partial definitions.",
        "- **Assessment Checkpoints:** Listen for accurate prior knowledge.",
        "- **Resources Needed:** Textbook opening page",
        f"- **Expected Outcome:** Students are curious about {chapter}.",
        "",
        "### Explore",
        f"- **Duration:** {explore} minutes",
        f"- **Teacher Activity:** Guide a hands-on or discussion-based exploration of {chapter}.",
        f"- **Student Activity:** Investigate examples from the textbook.",
        "- **Guiding Questions:** What patterns do you see?",
        "- **Expected Student Responses:** Observations and questions.",
        "- **Assessment Checkpoints:** Circulate and note misconceptions.",
        "- **Resources Needed:** Textbook, worksheets",
        "- **Expected Outcome:** Students form initial understanding.",
        "",
        "### Explain",
        f"- **Duration:** {explain} minutes",
        f"- **Teacher Activity:** Teach core concepts of {chapter} with worked examples.",
        f"- **Student Activity:** Take notes and ask clarifying questions.",
        "- **Guiding Questions:** How does this connect to what we explored?",
        "- **Expected Student Responses:** Definitions and explanations in own words.",
        "- **Assessment Checkpoints:** Quick oral checks.",
        "- **Resources Needed:** Board notes, textbook diagrams",
        "- **Expected Outcome:** Clear conceptual understanding.",
        "",
        "### Elaborate",
        f"- **Duration:** {elaborate} minutes",
        f"- **Teacher Activity:** Assign practice applying {chapter} to new situations.",
        f"- **Student Activity:** Solve problems or complete activities in groups.",
        "- **Guiding Questions:** Where else could we use this?",
        "- **Expected Student Responses:** Applied solutions and reasoning.",
        "- **Assessment Checkpoints:** Review group work.",
        "- **Resources Needed:** Practice sheet",
        "- **Expected Outcome:** Students apply knowledge independently.",
        "",
        "### Evaluate",
        f"- **Duration:** {evaluate} minutes",
        "- **Teacher Activity:** Conduct exit ticket and summarize.",
        "- **Student Activity:** Complete 2–3 assessment questions.",
        "- **Guiding Questions:** What was the most important idea today?",
        "- **Expected Student Responses:** Summary statements.",
        "- **Assessment Checkpoints:** Exit ticket collection.",
        "- **Resources Needed:** Exit slip",
        "- **Expected Outcome:** Learning verified; gaps identified.",
        "",
        "## Real-Life Connections",
        f"- Relate {chapter} to everyday experiences students can observe.",
        "",
        "## Common Misconceptions",
        "- Rushing through definitions without examples",
        "",
        "## Assessment",
        "- **Formative Assessment:** Oral questions during Explore and Explain",
        "- **Summative Assessment:** Short written check at end of class",
        "- **Exit Ticket:** 2 questions on today's key ideas",
        "- **Rubric:** 0–2 per question (incomplete / partial / complete)",
        "",
        "## Differentiated Instruction",
        "- **Support for Struggling Learners:** Sentence starters, paired work, simplified examples",
        "- **Support for Advanced Learners:** Extension problems and peer tutoring",
        "",
        "## Homework",
        f"- Re-read the {chapter} section and write 5 key terms with definitions.",
        "",
        "## Teacher Reflection",
        "- What went well? What needs re-teaching tomorrow?",
        "",
        "## Suggested Digital Resources",
        "- Videos: curriculum-aligned explainer on the topic",
        "- Interactive Activities: online quiz or simulation if available",
    ]
    return "\n".join(lines)


def _fallback_teaching_notes_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    objectives_raw = state.get("learning_objectives") or ""
    objectives = [o.strip() for o in objectives_raw.split("\n") if o.strip()] or [
        f"Understand key ideas in {chapter}"
    ]

    lines = [
        f"# Teaching Notes: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        "",
        "## Chapter at a Glance",
        f"This chapter introduces students to **{chapter}** in {subject}. "
        f"It builds foundational understanding teachers should reinforce through clear explanations and examples.",
        "",
        "---",
        "",
        "## Key Teaching Takeaways",
        *[f"- {obj}" for obj in objectives],
        "",
        "---",
        "",
        "## Essential Concepts to Emphasize",
        f"- Core ideas and vocabulary related to {chapter}",
        f"- Definitions and relationships students must internalize",
        f"- Common patterns or rules that appear throughout the chapter",
        "",
        "---",
        "",
        "## Teacher Preparation Guide",
        f"- Review the textbook section on {chapter} before class",
        "- Prepare 2–3 concrete examples (local or textbook-based)",
        "- Anticipate where students may get confused",
        "- Gather visuals or demonstrations if available",
        "",
        "---",
        "",
        "## Materials & Resources",
        "- Textbook",
        "- Whiteboard / projector",
        "- Charts, maps, or diagrams as relevant",
        "- Worksheets or handouts for guided practice",
        "",
        "---",
        "",
        "## Background Knowledge for Teachers",
        f"- Connect {chapter} to prior units in {subject}",
        f"- Understand why {chapter} matters beyond the classroom",
        "",
        "---",
        "",
        "## Student-Friendly Explanations",
        f"- Explain {chapter} using everyday analogies students can relate to",
        "- Break complex ideas into smaller steps",
        "- Use simple language before introducing formal terms",
        "",
        "---",
        "",
        "## Teaching Strategies",
        "- Start with prior knowledge activation",
        "- Use think-pair-share for key questions",
        "- Demonstrate with worked examples before independent practice",
        "- Encourage students to explain ideas in their own words",
        "",
        "---",
        "",
        "## Suggested Classroom Dialogue",
        "**Teacher:** What do you already know about this topic?",
        "**Student:** [Shares prior knowledge]",
        "**Teacher Follow-up:** That's a good start. Today we'll build on that by looking at…",
        "",
        "---",
        "",
        "## Questions to Stimulate Thinking",
        "- **Recall:** What are the main terms in this chapter?",
        "- **Understanding:** Can you explain this idea in your own words?",
        "- **Application:** Where would you use this in real life?",
        "- **Higher-order:** What would happen if…?",
        "",
        "---",
        "",
        "## Anticipated Student Responses",
        f"- Students may define {chapter} partially — prompt for examples",
        "- Strong students may connect to other subjects — acknowledge and extend",
        "",
        "---",
        "",
        "## Common Misconceptions & Remedies",
        "**Misconception:** Students memorize without understanding.",
        "**Why students think this:** Exam pressure or unclear explanations.",
        "**Correction Strategy:** Use counter-examples and ask students to justify answers.",
        "",
        "---",
        "",
        "## Difficult Areas and Scaffolding Strategies",
        f"- Identify the hardest sub-topic in {chapter}",
        "- Provide sentence frames and visual aids",
        "- Use guided practice before independent work",
        "",
        "---",
        "",
        "## Real-World Connections",
        f"- Relate {chapter} to daily life, local context, and current events where appropriate",
        "",
        "---",
        "",
        "## Cross-Curricular Connections",
        f"- Link {subject} concepts to literacy, numeracy, or other subjects when natural",
        "",
        "---",
        "",
        "## Differentiation Suggestions",
        "",
        "### Support for Struggling Learners",
        "- Pair with a peer; use visuals and simplified language",
        "- Provide partially completed examples",
        "",
        "### Extension for Advanced Learners",
        "- Challenge questions and enrichment reading",
        "- Ask students to teach a mini-lesson to the class",
        "",
        "---",
        "",
        "## Teaching Tips & Best Practices",
        "- Use visuals and frequent comprehension checks",
        "- Avoid rushing through definitions",
        "- Use local examples to increase engagement",
        "",
        "---",
        "",
        "## Interesting Facts & Extensions",
        f"- Share one surprising fact related to {chapter} to spark curiosity",
        "",
        "---",
        "",
        "## Digital Teaching Resources",
        "- Curriculum-aligned videos or simulations",
        "- Interactive quizzes for review",
        "",
        "---",
        "",
        "## Post-Lesson Reflection Questions",
        "- Which concepts did students struggle with most?",
        "- Which examples worked best?",
        "- What needs reinforcement in the next class?",
        "- Which misconceptions still remain?",
    ]
    return "\n".join(lines)


    return "\n".join(lines)


def _fallback_examples_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""

    lines = [
        f"# Examples: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        "",
        "## Concept Examples",
        f"**Example 1:** A basic illustration of the main idea in {chapter}.",
        f"- Explain how the core concept works using textbook language.",
        f"- Connect it to one simple scenario students already understand.",
        "",
        "## Worked Examples",
        f"**Problem:** Apply a key idea from {chapter}.",
        "**Step 1:** Identify what is given.",
        "**Step 2:** Choose the appropriate method or rule.",
        "**Step 3:** Solve and state the answer clearly.",
        "**Step 4:** Check whether the answer is reasonable.",
        "",
        "## Real-Life Examples",
        f"- Where students might notice {chapter} at home, school, or in the community.",
        f"- A news or daily-life situation that reflects the chapter theme.",
        "",
        "## Visual Examples",
        "- Diagram or chart from the textbook that clarifies the concept.",
        "- Sketch a simple model on the board to show relationships.",
        "",
        "## Guided Examples",
        f"**Prompt:** Try this {chapter} problem with a hint.",
        "**Hint:** Start by listing what you know.",
        "**Partial solution:** Set up the first step, then let students finish.",
        "",
        "## Challenge Examples",
        f"- A multi-step problem requiring students to combine two ideas from {chapter}.",
        "- An open-ended question: *What if the conditions changed?*",
        "",
        "## Common Mistakes in Examples",
        "**Mistake:** Jumping to the answer without showing reasoning.",
        "**Correct approach:** Write each step and justify it.",
        "",
        "**Mistake:** Using the wrong formula or definition.",
        f"**Correct approach:** Re-read the definition in the {chapter} section first.",
    ]
    return "\n".join(lines)


    return "\n".join(lines)


def _fallback_worksheet_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""

    lines = [
        f"# Worksheet: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        "",
        "**Name:** _________________________  **Date:** _________________________",
        "",
        "## Warm-Up Questions",
        "1. What do you already know about this topic?",
        "   Answer: _________________________________________________",
        "",
        f"2. Name one key term from {chapter}.",
        "   Answer: _________________________________________________",
        "",
        "## Practice Questions",
        f"3. Define the main idea of {chapter} in your own words.",
        "   Answer: _________________________________________________",
        "",
        "4. Give one example from the textbook.",
        "   Answer: _________________________________________________",
        "",
        "## Application Questions",
        f"5. How would you use what you learned about {chapter} in a real situation?",
        "   Answer: _________________________________________________",
        "",
        "## Higher-Order Thinking Questions",
        f"6. Compare two ideas from {chapter}. How are they similar and different?",
        "   Answer: _________________________________________________",
        "",
        "## Activity-Based Questions",
        f"7. Draw or describe a diagram that helps explain {chapter}.",
        "   Answer: _________________________________________________",
        "",
        "## Reflection Questions",
        "8. What was the most challenging part of this chapter for you? Why?",
        "   Answer: _________________________________________________",
    ]
    return "\n".join(lines)


    return "\n".join(lines)


def _fallback_quiz_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""

    def mcq(n: int, question: str, a: str, b: str, c: str, d: str, ans: str) -> list[str]:
        return [
            f"{n}. {question}",
            f"A. {a}",
            f"B. {b}",
            f"C. {c}",
            f"D. {d}",
            f"Answer: {ans}",
            "",
        ]

    blocks = [
        f"# Quiz: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        "Total Questions: 5",
        "",
        "## Multiple Choice Questions",
        "",
        *mcq(
            1,
            f"What is the main focus of {chapter}?",
            f"Understanding key ideas in {chapter}",
            "Unrelated topic A",
            "Unrelated topic B",
            "Unrelated topic C",
            "A",
        ),
        *mcq(
            2,
            f"Which subject does this chapter belong to?",
            "History only",
            subject,
            "Mathematics only",
            "Art only",
            "B",
        ),
        *mcq(
            3,
            f"Why is {chapter} important to study?",
            "It has no real use",
            "It builds understanding for later topics",
            "It is only for exams",
            "It replaces all other subjects",
            "B",
        ),
        *mcq(
            4,
            f"Which is the best way to learn {chapter}?",
            "Memorize without understanding",
            "Skip the textbook",
            "Read, practice, and apply examples",
            "Ignore class discussion",
            "C",
        ),
        *mcq(
            5,
            f"How can you apply ideas from {chapter} in daily life?",
            "By connecting concepts to real situations",
            "By avoiding the topic",
            "By copying answers only",
            "By never asking questions",
            "A",
        ),
        "## Answer Key",
        "",
        "1. A",
        "2. B",
        "3. B",
        "4. C",
        "5. A",
    ]
    return "\n".join(blocks)


def _fallback_homework_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""

    lines = [
        f"# Homework: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        "",
        "## Practice Tasks",
        f"- Review today's class notes on {chapter} and write a one-paragraph summary in your own words.",
        f"- Solve 3 textbook exercises related to {chapter} (show your working).",
        "",
        "## Real-Life Tasks",
        f"- Find one example of {chapter} in your home, neighborhood, or daily routine. Describe what you observed.",
        "",
        "## Observation Tasks",
        "- Spend 15 minutes observing something connected to today's lesson. Record 5 observations in a notebook.",
        "",
        "## Creative Tasks",
        f"- Create a poster, comic strip, or short skit that explains one key idea from {chapter}.",
        "",
        "## Mini Projects",
        f"- With household materials, build a simple model or diagram that represents a concept from {chapter}.",
        "",
        "## Research Tasks",
        f"- Read the {chapter} section in your textbook again and find 3 facts that surprised you. Write them down.",
        "",
        "## Reflection Tasks",
        "- What was the most interesting thing you learned today? What question do you still have?",
        "- How could you explain today's topic to a younger sibling or friend?",
    ]
    return "\n".join(lines)


    return "\n".join(lines)


def _fallback_ppt_outline_markdown(state: PlannerState) -> str:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    duration = int(state.get("duration_minutes") or 45)
    target = int(state.get("ppt_slide_count") or 12)
    objectives_raw = state.get("learning_objectives") or ""
    objectives = [o.strip() for o in objectives_raw.split("\n") if o.strip()] or [
        f"Understand key ideas in {chapter}"
    ]

    def slide(n: int, title: str, layout: str, side: str, bullets: list[str], notes: str, icon: str = "engage") -> list[str]:
        return [
            f"### Slide {n}: {title}",
            "",
            f"**Layout:** {layout}",
            f"**Side Heading:** {side}",
            f"**Icon:** {icon}",
            "**Slide Content:**",
            *[f"- {b}" for b in bullets],
            f"**Callout:** {bullets[0] if bullets else title}",
            f"**Speaker Notes:** {notes}",
            "",
            "---",
            "",
        ]

    pool = [
        slide(1, chapter, "title", subject or "Classroom Lesson", [], f"Welcome students and introduce {chapter}.", "engage"),
        slide(2, "What We Will Learn", "bullets", "Objectives", objectives[:4], "Connect each objective to today's activities.", "check"),
        slide(3, f"What is {chapter}?", "bullets", "Concept", ["Key definition in simple words", "Where we see it in daily life", "One important vocabulary word"], "Keep language concrete.", "diagram"),
        slide(4, "Concept vs Example", "two_column", "Compare", ["Main idea", "Important detail", "Common confusion"], "Work one example together.", "example"),
        slide(5, "How It Works", "steps", "Method", ["Notice the first clue", "Connect it to the rule", "Check with a quick example"], "Narrate each step aloud.", "try"),
        slide(6, "Why It Matters", "bullets", "Real life", ["Helps people make decisions", "Protects safety and resources", "Connects to our community"], "Ask for local examples.", "remember"),
        slide(7, "Try This", "bullets", "Practice", ["Quick practice question", "Talk with a partner", "Share one answer"], "Circulate and prompt.", "try"),
        slide(8, "What We Learned", "summary", "Recap", ["Main idea from today", "One tool or method we used", "One question still open"], "Exit ticket before leaving.", "remember"),
        slide(9, "Deeper Look", "bullets", "Extend", ["Another important detail", "Compare with an earlier idea", "Watch for this mistake"], "Stretch stronger learners.", "diagram"),
        slide(10, "Case Study", "two_column", "Apply", ["Situation / problem", "Data we notice", "Decision we make"], "Discuss in pairs.", "example"),
        slide(11, "Check Understanding", "bullets", "Check", ["Question 1", "Question 2", "Question 3"], "Use thumbs up / mini boards.", "check"),
        slide(12, "Remember This", "big_idea", "Big idea", [f"The big idea of {chapter} in one sentence"], "Have students restate it.", "remember"),
        slide(13, "Practice Round 2", "steps", "Practice", ["Read the prompt", "Choose a method", "Explain your answer"], "Cold-call gently.", "try"),
        slide(14, "Connect Forward", "bullets", "Next", ["Link to tomorrow's topic", "Home practice idea", "One curiosity question"], "Preview next class.", "engage"),
        slide(15, "Key Vocabulary", "two_column", "Words", ["Term 1 meaning", "Term 2 meaning"], "Choral read definitions.", "example"),
        slide(16, "Thank You", "summary", "Close", [f"Today we explored {chapter}", "Share one new idea", "See you next class"], "End with encouragement.", "remember"),
    ]

    # Rebuild numbering for the selected budget.
    chosen = pool[: max(6, min(target, 16))]
    blocks = [
        f"# Presentation Outline: {chapter}",
        "",
        f"Class: {grade}",
        f"Subject: {subject}",
        f"Chapter: {chapter}",
        f"Estimated Slides: {len(chosen)}",
        f"Estimated Duration: {duration} minutes",
        "",
        "## Presentation Overview",
        f"Students will explore **{chapter}** through clear visuals, examples, and short activities.",
        "",
        "---",
        "",
    ]
    for i, lines in enumerate(chosen, start=1):
        # Rewrite slide number in the first line.
        lines = list(lines)
        lines[0] = re.sub(r"^### Slide \d+:", f"### Slide {i}:", lines[0])
        blocks.extend(lines)
    return "\n".join(blocks)


def fallback_artifact(state: PlannerState, artifact_type: ArtifactType) -> dict[str, Any]:
    chapter = state.get("chapter_name") or "this chapter"
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    duration = int(state.get("duration_minutes") or 45)
    objectives_raw = state.get("learning_objectives") or ""
    objectives = [o.strip() for o in objectives_raw.split("\n") if o.strip()] or [
        f"Understand key ideas in {chapter}"
    ]

    if artifact_type == ArtifactType.LESSON_PLAN:
        return {
            "format": "markdown",
            "markdown": _fallback_lesson_plan_markdown(state),
        }

    if artifact_type == ArtifactType.TEACHING_NOTES:
        return {
            "format": "markdown",
            "markdown": _fallback_teaching_notes_markdown(state),
        }

    if artifact_type == ArtifactType.EXAMPLES:
        return {
            "format": "markdown",
            "markdown": _fallback_examples_markdown(state),
        }

    if artifact_type == ArtifactType.WORKSHEET:
        return {
            "format": "markdown",
            "markdown": _fallback_worksheet_markdown(state),
        }

    if artifact_type == ArtifactType.QUIZ:
        return {
            "format": "markdown",
            "markdown": _fallback_quiz_markdown(state),
        }

    if artifact_type == ArtifactType.HOMEWORK:
        return {
            "format": "markdown",
            "markdown": _fallback_homework_markdown(state),
        }

    if artifact_type == ArtifactType.PPT_OUTLINE:
        from app.services.lesson_planner.export.deck_schema import ppt_slides_from_markdown

        md = _fallback_ppt_outline_markdown(state)
        return {
            "format": "markdown",
            "markdown": md,
            "slides": ppt_slides_from_markdown(
                md,
                chapter=state.get("chapter_name"),
                subject=state.get("subject"),
                max_slides=int(state.get("ppt_slide_count") or 12),
            ),
            "template_id": state.get("ppt_template") or "clean_academic",
            "slide_count_target": int(state.get("ppt_slide_count") or 12),
            "chapter_name": state.get("chapter_name") or "",
            "subject": state.get("subject") or "",
            "figures": list(state.get("figures") or [])[:8],
        }

    return {}

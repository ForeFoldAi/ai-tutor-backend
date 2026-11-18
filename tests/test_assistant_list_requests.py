"""Ask Learn Mastro: "give me 50 questions" must produce the list, not a 3-item refusal."""

import pytest

from app.config import ASSISTANT_MAX_TOKENS
from app.modules.student_assistant.list_requests import plan_list_turn, requested_list


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Can you give me the 50 questions and answers for me to start the general knowledge?", (50, True)),
        ("I am interested in learning geography. Go ahead with listing 50 questions.", (50, True)),
        ("give me 20 MCQs on fractions", (20, True)),
        ("quiz me with fifty geography questions", (50, True)),
        ("list 10 facts about the moon", (10, False)),
        ("give me twenty fraction comparisons", (20, False)),
        ("Can you give me twenty more?", (20, False)),
        ("give me 20 more comparisons", (20, False)),
    ],
)
def test_detects_list_requests(query, expected):
    assert requested_list(query) == expected


@pytest.mark.parametrize(
    "query",
    ["explain photosynthesis in 5 sentences", "I got 20 questions wrong", "what is 50 + 20", "give me 3 examples"],
)
def test_ignores_non_list_messages(query):
    assert requested_list(query) is None


def test_twenty_more_inherits_questions_from_history():
    history = [
        {"role": "assistant", "content": "1. What is 2+2? — Answer: 4\n2. What is 3+3? — Answer: 6"},
    ]
    assert requested_list("Can you give me twenty more?", history) == (20, True)


def test_text_list_gets_room_for_every_item():
    plan = plan_list_turn("give me 50 gk questions", None, voice_mode=False)
    assert plan.expected_items == 50
    assert plan.max_tokens > ASSISTANT_MAX_TOKENS
    assert "Give all 50" in plan.prompt_block
    assert "no gaps" in plan.prompt_block
    full = "\n".join(f"{i}. Q{i}? — Answer: A{i}" for i in range(1, 51))
    assert not plan.list_incomplete(full)
    assert plan.list_incomplete("1. Q1? — Answer: A1\n2. Q2? — Answer: A2")


def test_list_incomplete_when_a_middle_number_is_missing():
    plan = plan_list_turn("give me 10 fraction comparisons", None, voice_mode=False)
    # 1..6, 8..10 — max is 10 but 7 is missing (the screenshot bug).
    gappy = "\n".join(
        f"{i}. {i}/2 ? {i}/4" for i in [1, 2, 3, 4, 5, 6, 8, 9, 10]
    )
    assert plan.list_incomplete(gappy)
    assert "contiguous" in plan.continue_prompt


def test_requests_over_the_cap_are_trimmed():
    plan = plan_list_turn("give me 100 gk questions", None, voice_mode=False)
    assert plan.expected_items == 50
    assert "first 50" in plan.prompt_block


def test_voice_runs_a_one_question_quiz():
    plan = plan_list_turn("give me 50 gk questions", None, voice_mode=True)
    assert "Question 1 of 50" in plan.prompt_block
    assert plan.max_tokens == ASSISTANT_MAX_TOKENS
    assert plan.expected_items == 0


def test_voice_quiz_continues_from_history_until_stopped():
    history = [
        {"role": "user", "content": "50 questions"},
        {"role": "assistant", "content": "Question 3 of 50: What is the capital of Japan?"},
    ]
    assert "Question 4 of 50" in plan_list_turn("Tokyo", history, voice_mode=True).prompt_block
    assert plan_list_turn("stop the quiz", history, voice_mode=True).prompt_block == ""
    assert plan_list_turn("Tokyo", history, voice_mode=False).prompt_block == ""
    last = [{"role": "assistant", "content": "Question 50 of 50: Which is the largest desert?"}]
    assert plan_list_turn("Sahara", last, voice_mode=True).prompt_block == ""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services import quiz_grading as qg

QUIZ_REPLY = """Here are your questions:
1. Who founded the Mughal empire?
a) Babur
b) Akbar
c) Humayun
2. Which ruler introduced the mansabdari system?
3. Name the dynasty that ruled from Thanjavur.
Answer when you're ready!"""


def _state():
    return qg.new_quiz_state(QUIZ_REPLY, "Babur founded the Mughal empire in 1526.")


def test_parse_questions_with_options():
    qs = qg.parse_quiz_questions(QUIZ_REPLY)
    assert len(qs) == 3
    assert qs[0].startswith("Who founded the Mughal empire?") and "a) Babur" in qs[0]


def test_parse_grades_maps_to_open_indexes():
    rows = "1: CORRECT | Babur\n2: PARTIALLY_CORRECT | Akbar\n3: nonsense"
    grades = qg.parse_grades(rows, [0, 2, 3])
    assert grades == [
        {"index": 0, "grade": "CORRECT", "note": "Babur"},
        {"index": 2, "grade": "PARTIAL", "note": "Akbar"},
    ]


def test_partial_credit_and_score():
    s = qg.apply_grades(_state(), [
        {"index": 0, "grade": "CORRECT", "note": ""},
        {"index": 1, "grade": "PARTIAL", "note": "Akbar"},
    ])
    assert qg.open_indexes(s) == [2]
    assert qg.quiz_score(s) == 1.5
    assert qg.spoken_score(1.5) == "1 and a half"


def test_second_miss_reveals_uncertain_costs_nothing():
    s = _state()
    s = qg.apply_grades(s, [{"index": 2, "grade": "UNCERTAIN", "note": ""}])
    assert s["items"][2]["attempts"] == 0 and not s["items"][2]["done"]
    s = qg.apply_grades(s, [{"index": 2, "grade": "INCORRECT", "note": ""}])
    assert s["items"][2]["attempts"] == 1 and not s["items"][2]["done"]
    s = qg.apply_grades(s, [{"index": 2, "grade": "INCORRECT", "note": "Cholas"}])
    assert s["items"][2]["done"] and s["items"][2]["revealed"]


def test_not_an_answer_changes_nothing():
    grades = [{"index": 0, "grade": "NOT_AN_ANSWER", "note": ""}]
    assert qg.is_not_an_answer(grades)
    s = qg.apply_grades(_state(), grades)
    assert qg.open_indexes(s) == [0, 1, 2]


def test_feedback_block_finished_quiz_says_score_and_no_new_question():
    s = _state()
    grades = [
        {"index": 0, "grade": "CORRECT", "note": "Babur"},
        {"index": 1, "grade": "PARTIAL", "note": "Akbar"},
        {"index": 2, "grade": "CORRECT", "note": "Cholas"},
    ]
    s = qg.apply_grades(s, grades)
    block = qg.build_feedback_block(s, grades)
    assert "2 and a half out of 3" in block
    assert "Do not ask a new quiz question" in block
    assert "Akbar" in block


def test_grade_quiz_answer_uses_llm_rows():
    with patch("app.services.llm_client.complete", new_callable=AsyncMock) as llm:
        llm.return_value = "1: CORRECT | Babur\n2: INCORRECT | Akbar"
        grades = asyncio.run(qg.grade_quiz_answer(_state(), "1-a, 2 Humayun"))
    assert [g["grade"] for g in grades] == ["CORRECT", "INCORRECT"]

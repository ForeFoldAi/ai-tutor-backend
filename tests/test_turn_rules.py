"""Turn rules from the Chapter 2 (Reshaping India's Political Map) replay failures."""
from app.services.chat_service.answer_types import _resolve_answer_type
from app.services.chat_service.turn_rules import (
    asks_for_short_answers,
    build_turn_rules,
    is_cite_request,
    is_non_answer,
    is_student_posed_question,
    quiz_count,
    requested_count,
)
from app.services.conversation_intent_classifier import FollowupType, classify_followup_regex


def test_requested_count():
    assert requested_count("explain the Delhi Sultanate in 3 sentences") == (3, "sentences")
    assert requested_count("tell me in two lines") == (2, "lines")
    assert requested_count("give it in 4 points") == (4, "points")
    assert requested_count("explain the Delhi Sultanate") is None


def test_quiz_count():
    assert quiz_count("ask me 5 questions") == 5
    assert quiz_count("give me two quick questions") == 2
    assert quiz_count("quiz me") is None
    assert quiz_count("ask me 50 questions") == 10


def test_short_preference():
    assert asks_for_short_answers("stop giving such long answers")
    assert asks_for_short_answers("keep it short")
    assert asks_for_short_answers("your answers are too long")
    assert not asks_for_short_answers("tell me about the long reign of Akbar")


def test_student_posed_question():
    assert is_student_posed_question("Which of the following was a Mughal emperor? Option A Babur Option B Ashoka")
    assert is_student_posed_question("True or false: the Cholas ruled Delhi")
    assert is_student_posed_question("a) Akbar b) Ashoka which one built Fatehpur Sikri")
    assert not is_student_posed_question("ask me 3 questions")


def test_cite_request():
    assert is_cite_request("prove it from my textbook")
    assert is_cite_request("where does it say that")
    assert not is_cite_request("show me the map in the textbook")


def test_non_answer():
    assert is_non_answer("you already asked that")
    assert is_non_answer("stop asking me questions")
    assert is_non_answer("why did the Mughal empire decline?")
    assert is_non_answer("stop giving such long answers")
    assert not is_non_answer("1-b, 2-a, 3-c")
    assert not is_non_answer("Akbar built Fatehpur Sikri")


def test_turn_rules_block():
    rules = build_turn_rules("explain it in 3 sentences")
    assert "exactly 3 sentences" in rules
    assert "FALSE STATEMENTS" in rules and "never say the context".lower() in rules.lower()
    short = build_turn_rules("what was the iqta system", prefers_short=True, voice=True)
    assert "under 40 words" in short
    mcq = build_turn_rules("True or false: Babur founded the Mughal empire")
    assert "STUDENT'S OWN QUESTION" in mcq
    new = build_turn_rules("teach me something new", explained_points=["delhi sultanate"])
    assert "delhi sultanate" in new and "NEW MATERIAL" in new


def test_answer_type_routing():
    history = [{"role": "assistant", "content": "The Mughals ruled from Delhi and Agra."}]
    assert _resolve_answer_type(
        "True or false: Akbar was the first Mughal emperor", subject_name="Social"
    ) == "answer_student_mcq"
    assert _resolve_answer_type(
        "prove it from my textbook", subject_name="Social", conversation_history=history
    ) == "cite"


def test_false_statements_are_claims_not_topics():
    from app.services.chapter_scope import is_student_statement

    assert is_student_statement("Indians became independent in 1956.")
    assert is_student_statement("Stats and Districts are exactly the same")
    assert is_student_statement("Every state in India has the same language.")
    assert not is_student_statement("Who invited the telephone?")
    assert not is_student_statement("who invented the telephone")
    assert not is_student_statement("tell me about the Mughals")
    assert not is_student_statement("I don't know the answer for this..")


def test_followup_phrasing_does_not_hide_a_new_topic():
    from app.services.chapter_scope import _should_skip_topic_scope_check

    q = "I don't understand why India needed state reorganization. Can you explain it with a simple example?"
    assert _should_skip_topic_scope_check(q) is False
    history = [{"role": "assistant", "content": "The Delhi Sultanate used the iqta system."}]
    assert _should_skip_topic_scope_check("can you explain it again", history) is True
    assert _should_skip_topic_scope_check("What were we discussing just now?", history) is True
    assert _should_skip_topic_scope_check("Can you explain the second point again?", history) is True


def test_praise_with_a_request_is_not_a_goodbye():
    from app.services.voice_ack import resolve_dialogue_act

    assert resolve_dialogue_act("I understood the explanation but can you give me a real life example?") is None
    assert resolve_dialogue_act("thank you, bye") == "closing"
    assert resolve_dialogue_act("ok") == "ack"


def test_classifier_regexes():
    assert classify_followup_regex("Option A Akbar Option B Babur which is right") == FollowupType.NEW_TOPIC
    assert classify_followup_regex("what did we cover today") == FollowupType.ASK_SUMMARY
    assert classify_followup_regex("in short") != FollowupType.ASK_SUMMARY

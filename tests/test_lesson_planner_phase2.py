from app.modules.teacher.lesson_planner.constants import ArtifactType
from app.services.lesson_planner.algorithms.bloom import BloomLevel, classify_bloom, classify_objectives
from app.services.lesson_planner.algorithms.concepts import extract_key_concepts
from app.services.lesson_planner.algorithms.curriculum import align_curriculum
from app.services.lesson_planner.algorithms.difficulty import estimate_difficulty
from app.services.lesson_planner.algorithms.misconceptions import detect_misconceptions
from app.services.lesson_planner.algorithms.pedagogy import enrich_pedagogy_metadata
from app.services.lesson_planner.algorithms.prerequisite_dag import build_prerequisite_dag
from app.services.lesson_planner.algorithms.worksheet_balance import balance_worksheet_questions
from app.services.lesson_planner.state import initial_state_from_payload


def test_bloom_classify_apply():
    assert classify_bloom("Students will solve word problems using fractions") == BloomLevel.APPLY


def test_classify_objectives_returns_levels():
    mapped = classify_objectives(["Define photosynthesis", "Explain the role of chlorophyll"])
    assert mapped[0]["bloom_level"] == "remember"
    assert mapped[1]["bloom_level"] == "understand"


def test_extract_key_concepts_from_text():
    text = "Photosynthesis converts light energy. Chlorophyll captures sunlight. Plants produce glucose."
    concepts = extract_key_concepts(text, top_k=5)
    assert concepts
    assert any("photosynthesis" in c for c in concepts)


def test_prerequisite_dag_topological_order():
    dag = build_prerequisite_dag(["fractions", "decimals", "percentages"])
    assert dag["topological_order"][0] == "fractions"
    assert len(dag["edges"]) == 2


def test_estimate_difficulty_in_range():
    score = estimate_difficulty("Analyze and compare two ecosystems", grade="CLASS_9")
    assert 1.0 <= score <= 5.0


def test_curriculum_alignment_band():
    aligned = align_curriculum(
        grade="CLASS_4",
        subject="Science",
        chapter_name="Plants",
        concepts=["photosynthesis", "leaves"],
    )
    assert aligned["curriculum_band"] == "primary"


def test_detect_misconceptions_from_context():
    text = "A common mistake is to confuse speed with velocity. Students often think heavier objects fall faster."
    items = detect_misconceptions(text)
    assert len(items) >= 1


def test_worksheet_balance_buckets():
    questions = [
        {"question": "Define atom."},
        {"question": "Solve the quadratic equation and justify your method."},
    ]
    balanced = balance_worksheet_questions(questions, grade="CLASS_10")
    assert "easy" in balanced
    assert balanced["distribution"]["easy"] >= 1


def test_enrich_pedagogy_metadata():
    state = initial_state_from_payload(
        {
            "grade": "CLASS_8",
            "subject": "Science",
            "chapter_name": "Force and Motion",
            "learning_objectives": "Explain Newton's first law",
            "requested_artifacts": [ArtifactType.LESSON_PLAN.value],
        },
        job_id="j",
        user_id="u",
        lesson_plan_id="p",
    )
    state["chapter_context"] = "Force causes acceleration. A common mistake is ignoring friction."
    meta = enrich_pedagogy_metadata(state)
    assert meta["key_concepts"]
    assert meta["prerequisite_dag"]["nodes"]
    assert meta["curriculum_alignment"]["subject"] == "Science"

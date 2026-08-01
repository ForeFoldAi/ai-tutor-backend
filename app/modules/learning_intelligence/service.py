"""LIA service layer."""

from __future__ import annotations

import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.models import (
    LiaConceptMastery,
    LiaKnowledgeGap,
    LiaMisconception,
)
from app.modules.learning_intelligence.schemas import (
    AffectedStudentInsight,
    ConceptMasteryOut,
    InterventionInsight,
    KnowledgeGapOut,
    KnowledgeMapOut,
    LearningEventIn,
    LearningEventResponse,
    MisconceptionOut,
    PredictionOut,
    StudentDigitalTwinOut,
    TeacherSummaryOut,
    TutorClassInsightsOut,
    TutorGuidanceObject,
    TutorGuidanceRequest,
    WeakTopicInsight,
)
from app.modules.users.models import User
from app.services.learning_intelligence.agents.memory import build_period_summary
from app.services.learning_intelligence.agents.student_modeling import (
    profile_to_dict,
    update_twin_from_events,
)
from app.services.learning_intelligence.agents.prediction import build_prediction_context
from app.services.learning_intelligence.orchestration.orchestrator import (
    get_teacher_summary_for_student,
    get_tutor_guidance,
    process_learning_event,
)
from app.modules.learning_intelligence.constants import PERIOD_WEEKLY


def ingest_event(db: Session, event: LearningEventIn) -> LearningEventResponse:
    return process_learning_event(db, event)


def sync_concept_graph(
    db: Session,
    *,
    board: str | None = None,
    class_level: str | None = None,
    subject_name: str | None = None,
) -> dict:
    from app.services.learning_intelligence.algorithms.concept_graph import seed_concept_graph

    result = seed_concept_graph(db, board=board, class_level=class_level, subject_name=subject_name)
    db.commit()
    return result


def fetch_tutor_guidance(db: Session, req: TutorGuidanceRequest) -> TutorGuidanceObject:
    return get_tutor_guidance(db, req)


def fetch_student_profile(db: Session, student_user_id: int) -> StudentDigitalTwinOut:
    profile = update_twin_from_events(db, student_user_id)
    return StudentDigitalTwinOut(**profile_to_dict(profile))


def fetch_knowledge_map(db: Session, student_user_id: int) -> KnowledgeMapOut:
    masteries = db.scalars(
        select(LiaConceptMastery).where(LiaConceptMastery.student_user_id == student_user_id)
    ).all()
    misconceptions = db.scalars(
        select(LiaMisconception).where(LiaMisconception.student_user_id == student_user_id).limit(20)
    ).all()
    gaps = db.scalars(
        select(LiaKnowledgeGap).where(LiaKnowledgeGap.student_user_id == student_user_id).limit(20)
    ).all()
    return KnowledgeMapOut(
        student_user_id=student_user_id,
        concepts=[
            ConceptMasteryOut(
                concept_key=m.concept_key,
                mastery_score=m.mastery_score,
                understanding_level=m.understanding_level,
                memorized_likelihood=m.memorized_likelihood,
            )
            for m in masteries
        ],
        misconceptions=[
            MisconceptionOut(
                concept_key=m.concept_key,
                misconception_key=m.misconception_key,
                description=m.description,
                confidence=m.confidence,
                occurrence_count=m.occurrence_count,
            )
            for m in misconceptions
        ],
        knowledge_gaps=[
            KnowledgeGapOut(
                concept_key=g.concept_key,
                gap_type=g.gap_type,
                missing_prerequisites=g.missing_prerequisites or [],
                confidence=g.confidence,
            )
            for g in gaps
        ],
    )


def fetch_teacher_summary(db: Session, student_user_id: int) -> TeacherSummaryOut:
    return get_teacher_summary_for_student(db, student_user_id)


def run_prediction(db: Session, student_user_id: int, concept_key: str | None = None) -> PredictionOut:
    prediction = build_prediction_context(db, student_user_id, concept_key)
    db.flush()
    return PredictionOut(
        prediction_type="risk_and_mastery",
        concept_key=concept_key,
        prediction=prediction,
        confidence=0.75,
    )


def refresh_weekly_summary(db: Session, student_user_id: int) -> dict:
    return build_period_summary(db, student_user_id, PERIOD_WEEKLY)


def _title_risk(level: str | None) -> str:
    key = (level or "").lower().replace("_", " ")
    return {
        "not started": "Not Started",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
    }.get(key, "Medium")


def _intervention_icon(text: str) -> str:
    t = text.lower()
    if "quiz" in t or "practice" in t:
        return "quiz"
    if "worksheet" in t or "homework" in t:
        return "worksheet"
    if "revision" in t or "session" in t or "revisit" in t or "recap" in t:
        return "revision"
    return "concept"


_BARE_SUBJECTS = frozenset(
    {
        "science",
        "social",
        "social science",
        "social studies",
        "math",
        "mathematics",
        "english",
        "general",
        "hindi",
        "evs",
    }
)
# Homework / project activity titles from lesson-planner artifacts — not curriculum topics.
_ACTIVITY_TITLE_RE = re.compile(
    r"\b("
    r"challenge|hunt|audit|invent|comic\s*strip|experiment|reflection|spotlight|"
    r"timeline|in the news|personal connection|build a|model of|emerging technologies|"
    r"weather watch|women in|household|scientific method reflection|scientific solution"
    r")\b",
    re.I,
)


def _is_meaningful_weak_topic(topic: str) -> bool:
    t = re.sub(r"\s+", " ", (topic or "").strip())
    if len(t) < 4 or len(t) > 72:
        return False
    low = t.lower()
    if low in _BARE_SUBJECTS:
        return False
    words = low.split()
    if len(words) >= 2 and words[0] == words[1]:
        return False  # "science science timeline ..."
    if _ACTIVITY_TITLE_RE.search(low):
        return False
    return True


def _chapter_progress_weak_topics(db: Session, student_ids: list[int]) -> list[WeakTopicInsight]:
    """Fallback when LIA concept graph is polluted / empty — use low chapter progress."""
    if not student_ids:
        return []
    from app.modules.student_learning.models import StudentChapterProgress

    topic_expr = func.coalesce(
        func.nullif(func.trim(StudentChapterProgress.chapter_name), ""),
        StudentChapterProgress.subject_name,
    )
    rows = db.execute(
        select(
            topic_expr.label("topic"),
            func.avg(StudentChapterProgress.progress_pct),
            func.count(func.distinct(StudentChapterProgress.user_id)),
        )
        .where(StudentChapterProgress.user_id.in_(student_ids))
        .group_by(topic_expr)
        .having(func.avg(StudentChapterProgress.progress_pct) < 55)
        .order_by(func.avg(StudentChapterProgress.progress_pct).asc())
        .limit(20)
    ).all()
    out: list[WeakTopicInsight] = []
    for topic, avg, student_count in rows:
        label = str(topic or "").strip() or "General"
        if not _is_meaningful_weak_topic(label):
            continue
        progress = int(round(float(avg or 0)))
        out.append(
            WeakTopicInsight(
                topic=label,
                struggle_percent=max(0, min(100, 100 - progress)),
                student_count=int(student_count or 1),
            )
        )
    return out


def fetch_class_insights(db: Session, tutor: User) -> TutorClassInsightsOut:
    from app.modules.auth.service import list_tutor_assigned_students
    from app.modules.learning_intelligence.models import LiaConcept
    from app.modules.school_admin.students.service import _class_fields
    from app.services.learning_intelligence.algorithms.concept_graph import humanize_concept_key

    students = list_tutor_assigned_students(db, tutor)
    student_ids = [s.id for s in students]
    topic_agg: dict[str, dict[str, int]] = {}
    affected: list[AffectedStudentInsight] = []
    interventions: list[InterventionInsight] = []
    seen_interventions: set[str] = set()

    seeded_labels = {
        c.concept_key: (c.display_name or c.chapter_name or humanize_concept_key(c.concept_key))
        for c in db.scalars(select(LiaConcept)).all()
    }

    for student in students:
        try:
            twin = fetch_student_profile(db, student.id)
            summary = fetch_teacher_summary(db, student.id)
            km = fetch_knowledge_map(db, student.id)
            risk = _title_risk(summary.risk_level or twin.risk_level)
            grade, _, _ = _class_fields(student)
            primary_weak = (
                summary.weaknesses[0]
                if summary.weaknesses
                else humanize_concept_key(twin.weak_concepts[0])
                if twin.weak_concepts
                else "General"
            )
            if not _is_meaningful_weak_topic(primary_weak):
                primary_weak = "General"

            if risk in ("High", "Medium"):
                affected.append(
                    AffectedStudentInsight(
                        student_user_id=student.id,
                        name=student.full_name,
                        grade=grade or "—",
                        topic=primary_weak,
                        risk_level=risk,
                    )
                )

            for concept in km.concepts:
                if concept.mastery_score >= 0.55:
                    continue
                # Prefer curriculum-seeded concepts; otherwise keep only meaningful labels.
                if concept.concept_key in seeded_labels:
                    topic = seeded_labels[concept.concept_key]
                else:
                    topic = humanize_concept_key(concept.concept_key)
                    if not _is_meaningful_weak_topic(topic):
                        continue
                bucket = topic_agg.setdefault(topic, {"struggle_sum": 0, "count": 0})
                bucket["struggle_sum"] += int((1 - concept.mastery_score) * 100)
                bucket["count"] += 1

            for gap in km.knowledge_gaps:
                if gap.concept_key in seeded_labels:
                    topic = seeded_labels[gap.concept_key]
                else:
                    topic = humanize_concept_key(gap.concept_key)
                    if not _is_meaningful_weak_topic(topic):
                        continue
                bucket = topic_agg.setdefault(topic, {"struggle_sum": 0, "count": 0})
                bucket["struggle_sum"] += int(gap.confidence * 100)
                bucket["count"] += 1

            for idx, rec in enumerate(summary.recommendations[:2]):
                key = rec.strip().lower()[:100]
                if not key or key in seen_interventions:
                    continue
                seen_interventions.add(key)
                priority = "High" if risk == "High" else "Medium" if risk == "Medium" else "Low"
                interventions.append(
                    InterventionInsight(
                        id=f"{student.id}-{idx}",
                        title=rec if len(rec) <= 72 else rec[:69] + "...",
                        description=rec,
                        priority=priority,
                        icon=_intervention_icon(rec),
                    )
                )
        except Exception:
            continue

    weak_topics = sorted(
        [
            WeakTopicInsight(
                topic=topic,
                struggle_percent=int(values["struggle_sum"] / max(values["count"], 1)),
                student_count=values["count"],
            )
            for topic, values in topic_agg.items()
            if _is_meaningful_weak_topic(topic)
        ],
        key=lambda item: (-item.student_count, -item.struggle_percent),
    )[:20]

    if not weak_topics:
        weak_topics = _chapter_progress_weak_topics(db, student_ids)

    affected.sort(key=lambda row: (0 if row.risk_level == "High" else 1, row.name.lower()))

    return TutorClassInsightsOut(
        weak_topics=weak_topics,
        affected_students=affected[:20],
        interventions=interventions[:15],
    )

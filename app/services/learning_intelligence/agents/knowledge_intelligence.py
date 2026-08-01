"""Knowledge Intelligence Agent — mastery, misconceptions, prerequisite gaps."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.constants import GAP_THRESHOLD
from app.modules.learning_intelligence.models import (
    LiaConceptMastery,
    LiaKnowledgeGap,
    LiaMisconception,
)
from app.modules.learning_intelligence.schemas import LearningEventIn
from app.services.learning_intelligence.algorithms.concept_graph import (
    dependency_edges,
    ensure_scope_seeded,
    transitive_prerequisites,
)
from app.services.learning_intelligence.algorithms.mastery import (
    concept_key_from_topic,
    find_missing_prerequisites,
    infer_understanding_level,
    memorized_likelihood,
    update_mastery_ema,
)

_MEMORIZATION_GAP_THRESHOLD = 0.65


def process_event_for_knowledge(db: Session, event: LearningEventIn) -> None:
    ensure_scope_seeded(
        db,
        subject_name=event.subject_name or "",
        board=event.payload.get("board", ""),
        class_level=event.payload.get("class_level", ""),
    )

    if event.event_type == "assignment_submitted":
        _process_assignment_items(db, event)
        return

    concept = event.concept_key or concept_key_from_topic(
        event.payload.get("topic", "") or event.payload.get("query", ""),
        subject=event.subject_name or "",
        chapter=event.chapter_name or "",
    )
    if not concept:
        return

    row = _get_or_create_mastery(db, event.student_user_id, concept)
    evidence = _evidence_type_from_event(event)
    row.mastery_score = update_mastery_ema(row.mastery_score, evidence)
    row.evidence_count += 1
    is_correct = event.payload.get("is_correct")
    row.understanding_level = infer_understanding_level(
        mastery_score=row.mastery_score,
        query=event.payload.get("query", "") or event.payload.get("question", ""),
        is_correct=is_correct if isinstance(is_correct, bool) else None,
        response_length=int(event.payload.get("response_words", 0)),
    )
    row.memorized_likelihood = memorized_likelihood(
        mastery_score=row.mastery_score,
        understanding_level=row.understanding_level,
        fast_correct=bool(event.payload.get("fast_correct")),
        slow_incorrect=bool(event.payload.get("slow_incorrect")),
    )
    row.last_evaluated_at = datetime.now(UTC)

    if evidence in ("incorrect", "confusion"):
        _upsert_misconception(db, event, concept)
    _update_knowledge_gaps(db, event.student_user_id, concept, row)


def _get_or_create_mastery(db: Session, student_user_id: int, concept: str) -> LiaConceptMastery:
    row = db.scalar(
        select(LiaConceptMastery).where(
            LiaConceptMastery.student_user_id == student_user_id,
            LiaConceptMastery.concept_key == concept,
        )
    )
    if row is None:
        row = LiaConceptMastery(
            student_user_id=student_user_id,
            concept_key=concept,
            last_evaluated_at=datetime.now(UTC),
        )
        db.add(row)
        db.flush()
    return row


def _process_assignment_items(db: Session, event: LearningEventIn) -> None:
    """Roll assignment evidence into curriculum concepts — not homework activity titles."""
    items = event.payload.get("items") or []
    if not isinstance(items, list):
        items = []

    subject = event.subject_name or ""
    chapter = (event.chapter_name or "").strip()

    # Prefer one chapter-level concept so project titles ("Science Timeline Challenge")
    # never become fake weak topics in class insights.
    if chapter:
        concept = concept_key_from_topic(chapter, subject=subject, chapter=chapter)
        row = _get_or_create_mastery(db, event.student_user_id, concept)
        scored = [i for i in items if isinstance(i, dict) and isinstance(i.get("is_correct"), bool)]
        if scored:
            correct = sum(1 for i in scored if i.get("is_correct") is True)
            evidence = "assignment_correct" if correct / len(scored) >= 0.6 else "assignment_incorrect"
        else:
            evidence = _evidence_type_from_event(event)
        row.mastery_score = update_mastery_ema(row.mastery_score, evidence)
        row.evidence_count += 1
        row.understanding_level = infer_understanding_level(
            mastery_score=row.mastery_score,
            query=chapter,
            is_correct=True if evidence == "assignment_correct" else False if evidence == "assignment_incorrect" else None,
        )
        row.memorized_likelihood = memorized_likelihood(
            mastery_score=row.mastery_score,
            understanding_level=row.understanding_level,
        )
        row.last_evaluated_at = datetime.now(UTC)
        if evidence == "assignment_incorrect":
            mini = LearningEventIn(
                event_type=event.event_type,
                student_user_id=event.student_user_id,
                subject_name=event.subject_name,
                chapter_name=event.chapter_name,
                concept_key=concept,
                payload={"query": chapter, "is_correct": False},
            )
            _upsert_misconception(db, mini, concept)
        _update_knowledge_gaps(db, event.student_user_id, concept, row)
        return

    # No chapter: only scored quiz/worksheet stems (skip open-ended homework titles).
    for item in items:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("is_correct"), bool):
            continue
        question = (item.get("question") or "").strip()[:120]
        if not question or len(question.split()) > 12:
            continue
        concept = concept_key_from_topic(question, subject=subject, chapter="")
        row = _get_or_create_mastery(db, event.student_user_id, concept)
        is_correct = item.get("is_correct")
        if is_correct is True:
            row.mastery_score = update_mastery_ema(row.mastery_score, "assignment_correct")
        else:
            row.mastery_score = update_mastery_ema(row.mastery_score, "assignment_incorrect")
            mini = LearningEventIn(
                event_type=event.event_type,
                student_user_id=event.student_user_id,
                subject_name=event.subject_name,
                chapter_name=event.chapter_name,
                concept_key=concept,
                payload={"query": question, "is_correct": False, "error_pattern": item.get("student_answer")},
            )
            _upsert_misconception(db, mini, concept)
        row.evidence_count += 1
        row.understanding_level = infer_understanding_level(
            mastery_score=row.mastery_score,
            query=question,
            is_correct=is_correct,
        )
        row.memorized_likelihood = memorized_likelihood(
            mastery_score=row.mastery_score,
            understanding_level=row.understanding_level,
            fast_correct=bool(is_correct) and float(event.payload.get("time_per_item_sec", 99)) < 8,
        )
        row.last_evaluated_at = datetime.now(UTC)
        _update_knowledge_gaps(db, event.student_user_id, concept, row)


def _evidence_type_from_event(event: LearningEventIn) -> str:
    p = event.payload
    if event.event_type == "quiz_answer":
        return "correct" if p.get("is_correct") else "incorrect"
    if p.get("is_correct") is True:
        return "correct"
    if p.get("is_correct") is False:
        return "incorrect"
    if float(p.get("confusion", 0)) >= 0.55:
        return "confusion"
    if p.get("hint_used"):
        return "hint_used"
    if event.event_type == "assignment_submitted":
        score = p.get("score")
        max_score = p.get("max_score")
        if score is not None and max_score:
            return "assignment_correct" if score / max_score >= 0.6 else "assignment_incorrect"
    if event.event_type == "chapter_completed":
        return "chapter_complete"
    if p.get("is_affirmation"):
        return "affirmation"
    return "correct" if float(p.get("understanding", 0)) >= 0.7 else "confusion"


def _misconception_key(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "unknown").lower())[:120]
    return slug or "unknown"


def _upsert_misconception(db: Session, event: LearningEventIn, concept: str) -> None:
    desc = (event.payload.get("error_pattern") or event.payload.get("query") or "recurring error")[:300]
    key = _misconception_key(desc)
    row = db.scalar(
        select(LiaMisconception).where(
            LiaMisconception.student_user_id == event.student_user_id,
            LiaMisconception.concept_key == concept,
            LiaMisconception.misconception_key == key,
        )
    )
    if row is None:
        row = LiaMisconception(
            student_user_id=event.student_user_id,
            concept_key=concept,
            misconception_key=key,
            description=desc,
            last_seen_at=datetime.now(UTC),
        )
        db.add(row)
    else:
        row.occurrence_count += 1
        row.confidence = min(0.95, row.confidence + 0.08)
        row.last_seen_at = datetime.now(UTC)


def _update_knowledge_gaps(
    db: Session,
    student_user_id: int,
    concept_key: str,
    mastery_row: LiaConceptMastery,
) -> None:
    masteries = db.scalars(
        select(LiaConceptMastery).where(LiaConceptMastery.student_user_id == student_user_id)
    ).all()
    mastery_map = {m.concept_key: m.mastery_score for m in masteries}
    current = mastery_map.get(concept_key, mastery_row.mastery_score)

    edges = dependency_edges(db)
    direct_missing = find_missing_prerequisites(concept_key, mastery_map, edges)
    transitive = transitive_prerequisites(concept_key, edges)
    weak_upstream = [p for p in transitive if mastery_map.get(p, 0.0) < GAP_THRESHOLD]
    missing = list(dict.fromkeys(direct_missing + weak_upstream))

    if mastery_row.memorized_likelihood >= _MEMORIZATION_GAP_THRESHOLD and current >= GAP_THRESHOLD:
        gap_type = "surface_memorization"
    elif missing:
        gap_type = "missing_prerequisite"
    elif current < GAP_THRESHOLD:
        gap_type = "incomplete"
    else:
        # Clear resolved gap
        row = db.scalar(
            select(LiaKnowledgeGap).where(
                LiaKnowledgeGap.student_user_id == student_user_id,
                LiaKnowledgeGap.concept_key == concept_key,
            )
        )
        if row is not None:
            db.delete(row)
        return

    row = db.scalar(
        select(LiaKnowledgeGap).where(
            LiaKnowledgeGap.student_user_id == student_user_id,
            LiaKnowledgeGap.concept_key == concept_key,
        )
    )
    if row is None:
        row = LiaKnowledgeGap(
            student_user_id=student_user_id,
            concept_key=concept_key,
            gap_type=gap_type,
            missing_prerequisites=missing,
            last_updated_at=datetime.now(UTC),
        )
        db.add(row)
    else:
        row.gap_type = gap_type
        row.missing_prerequisites = missing
        row.confidence = min(0.95, row.confidence + 0.05)
        row.last_updated_at = datetime.now(UTC)

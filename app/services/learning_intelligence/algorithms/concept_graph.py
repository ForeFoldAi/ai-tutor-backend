"""Curriculum concept graph — seed from catalog chapters + prerequisite chains."""

from __future__ import annotations

import re
from collections import defaultdict, deque

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.catalog.models import TextbookUpload
from app.modules.learning_intelligence.models import LiaConcept, LiaConceptDependency
from app.services.learning_intelligence.algorithms.mastery import concept_key_from_topic


def chapter_concept_key(*, subject_name: str, chapter_name: str, board: str = "", class_level: str = "") -> str:
    return concept_key_from_topic(chapter_name or "chapter", subject=subject_name, chapter=chapter_name)


def _chapter_sort_key(tb: TextbookUpload) -> tuple:
    ch = (tb.chapter or tb.file_name or "").strip()
    nums = re.findall(r"\d+", ch)
    num = int(nums[0]) if nums else tb.id
    return (num, ch.lower(), tb.id)


def seed_concept_graph(
    db: Session,
    *,
    board: str | None = None,
    class_level: str | None = None,
    subject_name: str | None = None,
) -> dict[str, int]:
    """Idempotent seed: concepts per textbook chapter + sequential chapter dependencies."""
    stmt = select(TextbookUpload).where(TextbookUpload.chapter.isnot(None))
    if board:
        stmt = stmt.where(TextbookUpload.board == board)
    if class_level:
        stmt = stmt.where(TextbookUpload.class_level == class_level)
    if subject_name:
        stmt = stmt.where(TextbookUpload.subject_name == subject_name)

    uploads = list(db.scalars(stmt).all())
    groups: dict[tuple[str, str, str], list[TextbookUpload]] = defaultdict(list)
    for tb in uploads:
        if not (tb.chapter or "").strip():
            continue
        key = (str(tb.board.value if hasattr(tb.board, "value") else tb.board), str(tb.class_level.value if hasattr(tb.class_level, "value") else tb.class_level), tb.subject_name)
        groups[key].append(tb)

    concepts_added = 0
    deps_added = 0

    for (_board, _class, subject), chapters in groups.items():
        ordered = sorted(chapters, key=_chapter_sort_key)
        prev_key: str | None = None
        for tb in ordered:
            ckey = chapter_concept_key(
                subject_name=subject,
                chapter_name=tb.chapter or "",
                board=_board,
                class_level=_class,
            )
            existing = db.scalar(select(LiaConcept).where(LiaConcept.concept_key == ckey))
            if existing is None:
                db.add(
                    LiaConcept(
                        concept_key=ckey,
                        display_name=tb.chapter or tb.file_name,
                        subject_name=subject,
                        chapter_name=tb.chapter,
                        metadata_json={"textbook_upload_id": tb.id, "board": _board, "class_level": _class},
                    )
                )
                concepts_added += 1
            if prev_key and prev_key != ckey:
                dep = db.scalar(
                    select(LiaConceptDependency).where(
                        LiaConceptDependency.prerequisite_key == prev_key,
                        LiaConceptDependency.dependent_key == ckey,
                    )
                )
                if dep is None:
                    db.add(
                        LiaConceptDependency(
                            prerequisite_key=prev_key,
                            dependent_key=ckey,
                            weight=1.0,
                        )
                    )
                    deps_added += 1
            prev_key = ckey

    db.flush()
    return {"concepts_added": concepts_added, "dependencies_added": deps_added, "groups": len(groups)}


def ensure_scope_seeded(
    db: Session,
    *,
    subject_name: str = "",
    board: str = "",
    class_level: str = "",
) -> None:
    """Lazy seed for the student's current subject scope."""
    if not subject_name:
        return
    count = db.scalar(select(LiaConcept.id).where(LiaConcept.subject_name == subject_name).limit(1))
    if count is not None:
        return
    seed_concept_graph(db, subject_name=subject_name)


def dependency_edges(db: Session) -> list[tuple[str, str, float]]:
    rows = db.scalars(select(LiaConceptDependency)).all()
    return [(r.prerequisite_key, r.dependent_key, r.weight) for r in rows]


def transitive_prerequisites(
    concept_key: str,
    edges: list[tuple[str, str, float]],
    *,
    max_depth: int = 4,
) -> list[str]:
    """BFS upstream prerequisites for a concept."""
    rev: dict[str, list[str]] = defaultdict(list)
    for prereq, dependent, _w in edges:
        rev[dependent].append(prereq)

    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(concept_key, 0)])
    out: list[str] = []

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for prereq in rev.get(node, []):
            if prereq in seen:
                continue
            seen.add(prereq)
            out.append(prereq)
            queue.append((prereq, depth + 1))
    return out


def humanize_concept_key(key: str) -> str:
    parts = key.split("|")
    label = parts[-1] if parts else key
    return label.replace("_", " ").strip() or key

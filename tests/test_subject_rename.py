"""Renaming a syllabus subject must carry its uploads, textbooks and vectors to the new name."""

from unittest.mock import patch

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from app.modules.catalog.models import (
    BoardEnum,
    ClassEnum,
    ProcessingStatusEnum,
    SubjectImage,
    SyllabusSubject,
    Textbook,
    TextbookImage,
    TextbookUpload,
)
from app.modules.catalog.router import update_syllabus
from app.modules.catalog.schemas import SyllabusUpdateRequest
import app.modules.schools.models  # noqa: F401  (registers "schools" for LiaLearningEvent's FK)
from app.modules.learning_intelligence.models import LiaLearningEvent
from app.modules.student_learning.models import StudentChapterProgress, StudentStudySession, StudentTutorChat


@compiles(JSONB, "sqlite")
def _jsonb_on_sqlite(*_args, **_kw):
    return "JSON"


OLD, NEW = "Mathematics", "Mathematics - Ganita Prakash Part - I"


def test_rename_moves_uploads_textbooks_progress_and_vectors():
    engine = create_engine("sqlite:///:memory:")
    for model in (SubjectImage, SyllabusSubject, TextbookUpload, Textbook, TextbookImage,
                  StudentStudySession, StudentChapterProgress, StudentTutorChat, LiaLearningEvent):
        model.__table__.create(engine)
    db = sessionmaker(bind=engine)()
    key = dict(board=BoardEnum.CBSE, class_level=ClassEnum.CLASS_8, subject_name=OLD)
    db.add_all([
        SyllabusSubject(id=1, **key),
        TextbookUpload(id=10, file_name="a.pdf", embedding_status=ProcessingStatusEnum.EMBEDDED, **key),
        TextbookUpload(id=11, file_name="b.pdf", board=BoardEnum.CBSE, class_level=ClassEnum.CLASS_9,
                       subject_name=OLD, embedding_status=ProcessingStatusEnum.EMBEDDED),
        Textbook(id=20, title="Book", file_name="b.pdf", file_path="x", status="COMPLETED", **key),
        TextbookImage(id=30, textbook_upload_id=10, file_name="f.png", subject=OLD),
        StudentChapterProgress(id=40, user_id=1, chapter_id=10, subject_name=OLD),
        StudentTutorChat(id=41, user_id=1, chapter_id=10, subject_name=OLD),
    ])
    db.commit()

    qdrant = QdrantClient(":memory:")
    old_coll, new_coll = "CBSE_CLASS_8_Mathematics", "CBSE_CLASS_8_Mathematics_-_Ganita_Prakash_Part_-_I"
    qdrant.create_collection(old_coll, vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE))
    qdrant.upsert(old_coll, points=[
        qm.PointStruct(id=i, vector=[1.0, 0.0, 0.0, float(i)], payload={"textbook_upload_id": "10"}) for i in range(300)
    ])

    with patch("app.services.vector_backend.use_qdrant", return_value=True), \
         patch("app.services.vector_backend.qdrant_common.get_qdrant_client", return_value=qdrant):
        update_syllabus(1, SyllabusUpdateRequest(subject_name=NEW), db, None)

    db.expire_all()
    assert db.get(TextbookUpload, 10).subject_name == NEW
    assert db.get(TextbookUpload, 11).subject_name == OLD  # other class untouched
    assert db.get(Textbook, 20).subject_name == NEW
    assert db.get(TextbookImage, 30).subject == NEW
    assert db.get(StudentChapterProgress, 40).subject_name == NEW
    assert db.get(StudentTutorChat, 41).subject_name == NEW
    assert not qdrant.collection_exists(old_coll)
    assert qdrant.count(new_coll).count == 300  # more than one scroll page
    assert qdrant.retrieve(new_coll, [7], with_vectors=True)[0].payload == {"textbook_upload_id": "10"}


def test_rename_blocked_while_upload_processing():
    engine = create_engine("sqlite:///:memory:")
    for model in (SubjectImage, SyllabusSubject, TextbookUpload, Textbook):
        model.__table__.create(engine)
    db = sessionmaker(bind=engine)()
    key = dict(board=BoardEnum.CBSE, class_level=ClassEnum.CLASS_8, subject_name=OLD)
    db.add_all([
        SyllabusSubject(id=1, **key),
        TextbookUpload(id=10, file_name="a.pdf", embedding_status=ProcessingStatusEnum.PROCESSING, **key),
    ])
    db.commit()

    try:
        update_syllabus(1, SyllabusUpdateRequest(subject_name=NEW), db, None)
        raise AssertionError("expected 409")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 409
    db.expire_all()
    assert db.get(SyllabusSubject, 1).subject_name == OLD
    assert db.get(TextbookUpload, 10).subject_name == OLD

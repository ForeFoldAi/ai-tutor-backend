"""Upload processing must not hold a pooled DB connection during long extraction/embedding."""

from sqlalchemy import Column, Integer, MetaData, Table, create_engine, event
from sqlalchemy.orm import sessionmaker

from app.modules.catalog.models import TextbookUpload
from app.modules.users.models import User  # noqa: F401  registers users for the FK
from app.services.catalog_pipeline.handlers import upload_handler


def test_no_connection_held_during_extraction(monkeypatch, tmp_path):
    engine = create_engine("sqlite://")
    meta = MetaData()
    Table("users", meta, Column("id", Integer, primary_key=True))
    TextbookUpload.__table__.to_metadata(meta)
    meta.create_all(engine)
    checked_out = []
    event.listen(engine, "checkout", lambda *a: checked_out.append(1))
    event.listen(engine, "checkin", lambda *a: checked_out.pop())

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    with engine.begin() as conn:
        conn.execute(meta.tables["textbook_uploads"].insert().values(
            id=1, board="STATE_BOARD", class_level="CLASS_5", subject_name="Maths", file_path=str(pdf), file_name="a.pdf",
        ))

    def fake_process(*_a, **_k):
        assert not checked_out, "DB connection held during extraction"
        return []

    import app.services.document_service as ds
    import app.services.vector_service as vs
    import app.services.image_service.textbook_image_extraction as img

    monkeypatch.setattr(upload_handler, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(upload_handler, "_local_path_for_upload", lambda p: p)
    monkeypatch.setattr(ds, "process_document", fake_process)
    monkeypatch.setattr(vs, "delete_collection_docs", lambda **_k: None)
    monkeypatch.setattr(img, "replace_all_images_after_reprocess", lambda *_a: 0)

    assert upload_handler.handle_upload_processing(1)["status"] == "completed"

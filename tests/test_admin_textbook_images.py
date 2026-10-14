"""Unit tests for Master Admin textbook image management endpoints."""

import io
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.modules.catalog.models import (
    BoardEnum,
    ClassEnum,
    TextbookImage,
    TextbookUpload,
)
from app.modules.catalog.schemas import (
    TextbookImageUpdateRequest,
)
from app.modules.catalog.router import (
    _serialize_textbook_image,
    list_upload_images,
    update_textbook_image,
    delete_textbook_image,
)
from app.modules.users.models import User
from app.modules.auth.constants import Role


def create_test_db():
    engine = create_engine("sqlite:///:memory:")
    TextbookUpload.__table__.create(engine)
    TextbookImage.__table__.create(engine)
    return sessionmaker(bind=engine)()


def test_serialize_textbook_image():
    im = TextbookImage(
        id=42,
        textbook_upload_id=10,
        page_index=3,
        sequence=1,
        file_name="figures/fig_2_1.png",
        caption="Structure of a plant cell",
        caption_normalized="structure of a plant cell",
        image_type="diagram",
        figure_number="Fig. 2.1",
        title="Plant Cell",
        educational_role="primary_concept",
        is_decorative=False,
    )
    serialized = _serialize_textbook_image(im)
    assert serialized.id == 42
    assert serialized.textbook_upload_id == 10
    assert serialized.page_index == 3
    assert serialized.image_url == "/auth/catalog/textbook-images/10/figures/fig_2_1.png"
    assert serialized.caption == "Structure of a plant cell"
    assert serialized.figure_number == "Fig. 2.1"
    assert serialized.image_type == "diagram"


def test_update_and_delete_textbook_image():
    db = create_test_db()
    admin_user = User(id=1, email="admin@test.com", role=Role.MASTER_ADMIN)

    upload = TextbookUpload(
        id=1,
        file_name="ch1.pdf",
        board=BoardEnum.CBSE,
        class_level=ClassEnum.CLASS_8,
        subject_name="Science",
        file_path="CBSE/CLASS_8/Science/ch1.pdf",
    )
    db.add(upload)
    db.commit()

    im = TextbookImage(
        id=101,
        textbook_upload_id=1,
        page_index=2,
        sequence=0,
        file_name="figures/fig_1.png",
        caption="Old caption",
        image_type="unknown",
        educational_role="unknown",
        is_decorative=False,
    )
    db.add(im)
    db.commit()

    # List images
    imgs = list_upload_images(upload_id=1, db=db, _current_user=admin_user)
    assert len(imgs) == 1
    assert imgs[0].id == 101

    # Update image
    update_req = TextbookImageUpdateRequest(
        caption="Fig. 1.2: Photosynthesis Process",
        figure_number="Fig. 1.2",
        image_type="process",
        educational_role="primary_concept",
        is_decorative=False,
    )
    updated = update_textbook_image(
        image_id=101, payload=update_req, db=db, _current_user=admin_user
    )
    assert updated.caption == "Fig. 1.2: Photosynthesis Process"
    assert updated.figure_number == "Fig. 1.2"
    assert updated.image_type == "process"
    assert updated.caption_normalized is not None

    # Delete image
    del_resp = delete_textbook_image(
        image_id=101, db=db, _current_user=admin_user
    )
    assert del_resp.message == "Image deleted successfully."

    # Verify deleted
    imgs_after = list_upload_images(upload_id=1, db=db, _current_user=admin_user)
    assert len(imgs_after) == 0

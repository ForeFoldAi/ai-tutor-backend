"""HTTP routes for ephemeral tutor image upload / analyze."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import IMAGE_MAX_SIZE_MB
from app.modules.auth.dependencies import get_current_user
from app.modules.image_understanding.errors import ImageUnderstandingError
from app.modules.image_understanding.schemas import (
    ImageAnalyzeRequest,
    ImageAnalyzeResponse,
    ImageUploadResponse,
)
from app.modules.image_understanding import service as iu_service
from app.modules.users.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/tutor/images", tags=["tutor-images"])


@router.post("", response_model=ImageUploadResponse)
async def upload_tutor_image(
    current_user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    max_bytes = int(IMAGE_MAX_SIZE_MB * 1024 * 1024)
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"That image is too large. Please upload an image under {IMAGE_MAX_SIZE_MB:g} MB.",
        )
    try:
        meta = await iu_service.save_upload(
            raw,
            user_id=int(current_user.id),
            filename=file.filename,
            declared_mime=file.content_type,
        )
    except ImageUnderstandingError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from None
    except Exception:
        logger.exception("tutor image upload failed")
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid JPG, PNG, or WEBP image.",
        ) from None

    return ImageUploadResponse(
        image_id=meta.image_id,
        content_hash=meta.content_hash,
        expires_at=meta.expires_at,
        mime_type=meta.mime_type,
        width=meta.width,
        height=meta.height,
    )


@router.post("/analyze", response_model=ImageAnalyzeResponse)
async def analyze_tutor_images(
    payload: ImageAnalyzeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    try:
        bundle = await iu_service.understand(
            payload.image_ids,
            payload.message,
            user_id=int(current_user.id),
            class_level=payload.class_level,
            subject_name=payload.subject_name,
            board=payload.board,
            delete_after=False,
        )
    except ImageUnderstandingError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from None
    if bundle is None:
        raise HTTPException(status_code=400, detail="Please upload an image first.")
    return ImageAnalyzeResponse(
        result=bundle.result,
        intent=bundle.intent,
        retrieval_query=bundle.retrieval_query,
        low_confidence=bundle.low_confidence,
    )

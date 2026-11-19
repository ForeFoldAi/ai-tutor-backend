from __future__ import annotations

from fastapi import APIRouter

from app.modules.school_admin.classes.router import router as classes_router
from app.modules.school_admin.credentials.router import router as credentials_router
from app.modules.school_admin.dashboard.router import router as dashboard_router
from app.modules.school_admin.settings.router import router as settings_router
from app.modules.school_admin.students.legacy_router import router as students_legacy_router
from app.modules.school_admin.students.router import router as students_router
from app.modules.school_admin.teachers.legacy_router import router as teachers_legacy_router
from app.modules.school_admin.teachers.router import router as teachers_router

router = APIRouter()
router.include_router(classes_router)
router.include_router(teachers_router)
router.include_router(teachers_legacy_router)
router.include_router(settings_router)
router.include_router(students_router)
router.include_router(students_legacy_router)
router.include_router(dashboard_router)
router.include_router(credentials_router)

__all__ = ["router"]

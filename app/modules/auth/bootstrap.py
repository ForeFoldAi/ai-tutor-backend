import logging
import os

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.modules.auth.constants import Role
from app.modules.organizations.models import Organization
from app.modules.schools.models import School
from app.modules.users.models import User

logger = logging.getLogger("auth_bootstrap")


def _env(*keys: str, default: str) -> str:
    for key in keys:
        val = os.environ.get(key)
        if val:
            return val
    return default


def _ensure_user(
    db,
    *,
    full_name: str,
    username: str,
    password: str,
    role: Role,
    organization_id=None,
    school_id=None,
    teaching_board: str | None = None,
    teaching_classes: list[dict] | None = None,
) -> bool:
    email = f"{username.lower()}@example.com"
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return False
    db.add(
        User(
            full_name=full_name,
            email=email,
            password_hash=hash_password(password),
            role=role,
            is_active=True,
            is_verified=True,
            organization_id=organization_id,
            school_id=school_id,
            teaching_board=teaching_board if role in (Role.TUTOR, Role.STUDENT) else None,
            teaching_classes=teaching_classes if role in (Role.TUTOR, Role.STUDENT) else None,
        )
    )
    return True


def seed_test_users_if_missing() -> None:
    """
    Creates default test accounts (if absent) so frontend demo creds can be
    used against backend /auth/login as well.
    """
    db = SessionLocal()
    try:
        org = db.query(Organization).filter(Organization.name == "Testing Organization").first()
        if not org:
            org = Organization(name="Testing Organization", phone="+910000000000", address="Test")
            db.add(org)
            db.flush()

        school = db.query(School).filter(School.name == "Testing School").first()
        if not school:
            school = School(organization_id=org.id, name="Testing School", board="CBSE")
            db.add(school)
            db.flush()

        student_username = _env("VITE_TEST_STUDENT_USERNAME", "TEST_STUDENT_USERNAME", default="student")
        tutor_username = _env("VITE_TEST_TUTOR_USERNAME", "TEST_TUTOR_USERNAME", default="tutor")
        school_admin_username = _env(
            "VITE_TEST_SCHOOL_ADMIN_USERNAME",
            "TEST_SCHOOL_ADMIN_USERNAME",
            default="admin",
        )
        master_username = _env("VITE_TEST_MASTER_ADMIN_USERNAME", "TEST_MASTER_ADMIN_USERNAME", default="master")
        org_admin_username = _env("VITE_TEST_ORG_ADMIN_USERNAME", "TEST_ORG_ADMIN_USERNAME", default="organization")

        student_password = _env("VITE_TEST_STUDENT_PASSWORD", "TEST_STUDENT_PASSWORD", default="password")
        tutor_password = _env("VITE_TEST_TUTOR_PASSWORD", "TEST_TUTOR_PASSWORD", default="password")
        school_admin_password = _env(
            "VITE_TEST_SCHOOL_ADMIN_PASSWORD",
            "TEST_SCHOOL_ADMIN_PASSWORD",
            default="password",
        )
        master_password = _env("VITE_TEST_MASTER_ADMIN_PASSWORD", "TEST_MASTER_ADMIN_PASSWORD", default="password")
        org_admin_password = _env("VITE_TEST_ORG_ADMIN_PASSWORD", "TEST_ORG_ADMIN_PASSWORD", default="password")

        created: list[str] = []
        if _ensure_user(
            db,
            full_name="Student User",
            username=student_username,
            password=student_password,
            role=Role.STUDENT,
            organization_id=org.id,
            school_id=school.id,
            teaching_board="CBSE",
            teaching_classes=[{"grade": "9", "sections": ["A"]}],
        ):
            created.append(f"{Role.STUDENT.value}: {student_username.lower()}@example.com")
        if _ensure_user(
            db,
            full_name="Tutor User",
            username=tutor_username,
            password=tutor_password,
            role=Role.TUTOR,
            organization_id=org.id,
            school_id=school.id,
            teaching_board="CBSE",
            teaching_classes=[
                {"grade": "9", "sections": ["A", "B"]},
                {"grade": "10", "sections": ["A"]},
            ],
        ):
            created.append(f"{Role.TUTOR.value}: {tutor_username.lower()}@example.com")
        if _ensure_user(
            db,
            full_name="School Admin User",
            username=school_admin_username,
            password=school_admin_password,
            role=Role.SCHOOL_ADMIN,
            organization_id=org.id,
            school_id=school.id,
        ):
            created.append(f"{Role.SCHOOL_ADMIN.value}: {school_admin_username.lower()}@example.com")
        if _ensure_user(
            db,
            full_name="Master Admin User",
            username=master_username,
            password=master_password,
            role=Role.MASTER_ADMIN,
        ):
            created.append(f"{Role.MASTER_ADMIN.value}: {master_username.lower()}@example.com")
        if _ensure_user(
            db,
            full_name="Organization Admin User",
            username=org_admin_username,
            password=org_admin_password,
            role=Role.ORG_ADMIN,
            organization_id=org.id,
        ):
            created.append(f"{Role.ORG_ADMIN.value}: {org_admin_username.lower()}@example.com")
        db.commit()
        if created:
            logger.info("Created demo users: %s", "; ".join(created))
        else:
            logger.info("All demo role users already present; no inserts.")
    except Exception as e:
        db.rollback()
        logger.warning("Skipping test-user bootstrap: %s", e)
    finally:
        db.close()

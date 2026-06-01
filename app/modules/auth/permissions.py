from app.modules.auth.constants import Role

ADMIN_CREATE_MATRIX: dict[Role, set[Role]] = {
    Role.MASTER_ADMIN: {Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.TUTOR, Role.STUDENT, Role.MASTER_ADMIN},
    Role.ORG_ADMIN: {Role.SCHOOL_ADMIN, Role.TUTOR, Role.STUDENT},
    Role.SCHOOL_ADMIN: {Role.TUTOR, Role.STUDENT},
    Role.TUTOR: {Role.STUDENT},
}

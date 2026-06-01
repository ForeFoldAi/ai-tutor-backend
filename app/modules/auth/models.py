"""
Compatibility module to keep auth domain discoverable.
Primary SQLAlchemy models live in domain-focused modules:
- users.models
- organizations.models
- schools.models
- sessions.models
"""

from app.modules.organizations.models import Organization
from app.modules.schools.models import School
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User

__all__ = ["Organization", "School", "SessionToken", "User"]

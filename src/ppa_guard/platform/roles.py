"""Five roles and their permissions. This table is the single source of truth for authorization.

Scope qualifiers are enforced in services (authz.py / gateway.py), not here:
  *_own   -> only applications the user owns (Developer)
  *_assigned -> only applications the user is assigned to
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    ADMIN = "platform_admin"
    ANALYST = "security_analyst"
    DEVELOPER = "developer"
    AUDITOR = "auditor"
    MEMBER = "member"


ROLE_LABELS = {
    Role.ADMIN: "Platform Administrator",
    Role.ANALYST: "Security Analyst",
    Role.DEVELOPER: "Application Developer / Integrator",
    Role.AUDITOR: "Auditor",
    Role.MEMBER: "Member",
}

# permission -> roles that hold it
PERMISSIONS: dict[str, set[Role]] = {
    # scanning (application scope applied in gateway.py)
    "scan.run.any_app":       {Role.ADMIN, Role.ANALYST},
    "scan.run.assigned_app":  {Role.DEVELOPER, Role.MEMBER},
    # viewing scan records (metadata; never raw prompts unless retention is on)
    "scan.view.org":          {Role.ADMIN, Role.ANALYST, Role.AUDITOR},
    "scan.view.own_apps":     {Role.DEVELOPER},
    "scan.view.self":         {Role.ADMIN, Role.ANALYST, Role.DEVELOPER, Role.AUDITOR, Role.MEMBER},
    "scan.view.retained_text": {Role.ADMIN, Role.ANALYST},
    # WARN review workflow
    "review.view":            {Role.ADMIN, Role.ANALYST},
    "review.decide":          {Role.ADMIN, Role.ANALYST},
    # applications, policies, integrations
    "app.view.org":           {Role.ADMIN, Role.ANALYST, Role.AUDITOR},
    "app.create":             {Role.ADMIN, Role.DEVELOPER},
    "app.manage.any":         {Role.ADMIN},
    "app.manage.own":         {Role.DEVELOPER},
    # members and roles
    "members.view":           {Role.ADMIN, Role.AUDITOR},
    "members.manage":         {Role.ADMIN},
    # audit
    "audit.view":             {Role.ADMIN, Role.ANALYST, Role.AUDITOR},
    "audit.verify":           {Role.ADMIN, Role.AUDITOR},
    # system status
    "system.view":            {Role.ADMIN, Role.ANALYST, Role.DEVELOPER, Role.AUDITOR, Role.MEMBER},
}


def has(role: Role | str, permission: str) -> bool:
    if permission not in PERMISSIONS:
        raise KeyError(f"unknown permission {permission!r}")
    return Role(role) in PERMISSIONS[permission]

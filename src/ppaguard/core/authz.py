"""Server-side authorization. Every service function calls into here; hiding a button is never the control.

Rule 2 (authorize every action): `require()` on every operation.
Rule 3 (organization isolation): the org comes from the caller's membership, re-read from the database on every
         request; objects from another org raise NotFound (indistinguishable from a missing id).
Rule 4 (application scope): `app_for()` returns an application only inside the caller's scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.engine import Connection

from .db import app_members, applications, memberships, organizations, scans, users
from .roles import Role, has


class AuthError(Exception):
    pass


class NotAuthenticated(AuthError):
    pass


class Forbidden(AuthError):
    pass


class NotFound(AuthError):
    pass


@dataclass(frozen=True)
class Principal:
    kind: str                 # "user" | "api_key"
    org_id: str
    role: Role
    user_id: str | None = None
    email: str | None = None
    app_id: str | None = None          # api keys are bound to one application
    api_key_id: str | None = None
    label: str = ""

    @property
    def actor(self) -> str:
        return self.email if self.kind == "user" else f"api-key:{self.label}"


def memberships_for(conn: Connection, email: str) -> list[dict]:
    q = (select(memberships.c.org_id, memberships.c.role, organizations.c.name, organizations.c.industry)
         .join(users, users.c.id == memberships.c.user_id).join(organizations, organizations.c.id == memberships.c.org_id)
         .where(users.c.email == email.lower(), users.c.status == "active", memberships.c.status == "active")
         .order_by(organizations.c.name))
    return [dict(r) for r in conn.execute(q).mappings()]


def load_principal(conn: Connection, email: str | None, org_id: str | None = None) -> Principal:
    """Rule 1 + re-check on every request: an unknown, disabled or non-member identity gets nothing."""
    if not email:
        raise NotAuthenticated("not signed in")
    u = conn.execute(select(users).where(users.c.email == email.lower())).mappings().first()
    if u is None:
        raise Forbidden("this account has not been invited to PPA Guard")
    if u["status"] != "active":
        raise Forbidden("this account is disabled")
    ms = memberships_for(conn, email)
    if not ms:
        raise Forbidden("this account has no active organization membership")
    m = next((x for x in ms if x["org_id"] == org_id), None) if org_id else ms[0]
    if m is None:
        raise Forbidden("not a member of the requested organization")
    return Principal("user", m["org_id"], Role(m["role"]), user_id=u["id"], email=u["email"], label=u["display_name"])


def require(p: Principal, permission: str) -> None:
    if not has(p.role, permission):
        raise Forbidden(f"{p.role.value} lacks permission {permission}")


def _assigned(conn: Connection, p: Principal, app_id: str) -> bool:
    return conn.execute(select(app_members.c.app_id).where(app_members.c.app_id == app_id,
                                                           app_members.c.user_id == p.user_id)).first() is not None


def app_for(conn: Connection, p: Principal, app_id: str, purpose: str) -> dict:
    """purpose: view | scan | manage. Raises NotFound outside the org or visible scope, Forbidden inside it."""
    app = conn.execute(select(applications).where(applications.c.id == app_id,
                                                  applications.c.org_id == p.org_id)).mappings().first()
    if app is None:
        raise NotFound("application not found")
    app = dict(app)
    if p.kind == "api_key":
        if app_id != p.app_id or purpose != "scan":
            raise NotFound("application not found")
        if app["status"] != "active":
            raise Forbidden("application is paused")
        return app
    owner = app["owner_user_id"] == p.user_id
    visible = has(p.role, "app.view.org") or owner or _assigned(conn, p, app_id)
    if not visible:
        raise NotFound("application not found")
    if purpose == "view":
        return app
    if purpose == "scan":
        if app["status"] != "active":
            raise Forbidden("application is paused")
        if has(p.role, "scan.run.any_app"):
            return app
        if has(p.role, "scan.run.assigned_app") and (owner or _assigned(conn, p, app_id)):
            return app
        raise Forbidden("not allowed to run scans in this application")
    if purpose == "manage":
        if has(p.role, "app.manage.any") or (has(p.role, "app.manage.own") and owner):
            return app
        raise Forbidden("not allowed to manage this application")
    raise ValueError(purpose)


def apps_visible(conn: Connection, p: Principal, purpose: str = "view") -> list[dict]:
    rows = conn.execute(select(applications).where(applications.c.org_id == p.org_id)
                        .order_by(applications.c.name)).mappings().all()
    out = []
    for r in rows:
        try:
            out.append(app_for(conn, p, r["id"], purpose))
        except AuthError:
            pass
    return out


def scan_scope(conn: Connection, p: Principal):
    """WHERE clause limiting scan records to what this principal may see."""
    base = scans.c.org_id == p.org_id
    if has(p.role, "scan.view.org"):
        return base
    if has(p.role, "scan.view.own_apps"):
        own = [a["id"] for a in apps_visible(conn, p)]
        return and_(base, or_(scans.c.app_id.in_(own), scans.c.user_id == p.user_id))
    return and_(base, scans.c.user_id == p.user_id)

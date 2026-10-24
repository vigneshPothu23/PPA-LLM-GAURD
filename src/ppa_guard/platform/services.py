"""Administrative and read operations. Each one authorizes first (Rule 2) and scopes to the caller's org (Rule 3).
Every change writes an audit event in the same transaction.
"""

from __future__ import annotations

import json
import secrets
from datetime import timedelta

from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from . import audit, policies
from .authz import Forbidden, NotFound, Principal, app_for, apps_visible, require, scan_scope
from .db import (api_keys, app_members, applications, audit_events, memberships, new_id, now, organizations,
                 retained_inputs, reviews, scans, users)
from .gateway import hash_key
from .roles import Role, has


class Services:
    def __init__(self, engine: Engine):
        self.engine = engine

    # ------------------------------------------------------------------ organization / members
    def create_organization(self, name: str, industry: str, admin_email: str, admin_name: str) -> str:
        """Bootstrap only (no principal yet): creates an org and its first Platform Administrator."""
        if industry not in policies.PROFILES:
            raise ValueError("unknown industry profile")
        with self.engine.begin() as conn:
            org_id = new_id()
            conn.execute(insert(organizations).values(id=org_id, name=name, industry=industry, created_at=now()))
            uid = self._ensure_user(conn, admin_email, admin_name)
            conn.execute(insert(memberships).values(user_id=uid, org_id=org_id, role=Role.ADMIN.value, created_at=now()))
            audit.append(conn, org_id, "system", "org.created", f"org:{org_id}", {"name": name, "industry": industry,
                                                                                "first_admin": admin_email.lower()})
        return org_id

    @staticmethod
    def _ensure_user(conn, email: str, name: str) -> str:
        u = conn.execute(select(users.c.id).where(users.c.email == email.lower())).first()
        if u:
            return u[0]
        uid = new_id()
        conn.execute(insert(users).values(id=uid, email=email.lower(), display_name=name or email, status="active", created_at=now()))
        return uid

    def list_members(self, p: Principal) -> list[dict]:
        require(p, "members.view")
        with self.engine.connect() as conn:
            q = (select(users.c.id, users.c.email, users.c.display_name, memberships.c.status, memberships.c.role)
                 .join(memberships, memberships.c.user_id == users.c.id).where(memberships.c.org_id == p.org_id)
                 .order_by(users.c.display_name))
            return [dict(r) for r in conn.execute(q).mappings()]

    def invite_member(self, p: Principal, email: str, name: str, role: Role) -> str:
        require(p, "members.manage")
        email = (email or "").strip().lower()
        if "@" not in email:
            raise ValueError("a valid email address is required")
        with self.engine.begin() as conn:
            uid = self._ensure_user(conn, email, name)
            if conn.execute(select(memberships).where(memberships.c.user_id == uid, memberships.c.org_id == p.org_id)).first():
                raise ValueError("already a member of this organization")
            conn.execute(insert(memberships).values(user_id=uid, org_id=p.org_id, role=Role(role).value, created_at=now()))
            audit.append(conn, p.org_id, p.actor, "member.invited", f"user:{email}", {"role": Role(role).value})
        return uid

    def _admins(self, conn, org_id):
        return conn.execute(select(func.count()).select_from(memberships).join(users, users.c.id == memberships.c.user_id)
                            .where(memberships.c.org_id == org_id, memberships.c.role == Role.ADMIN.value,
                                   memberships.c.status == "active", users.c.status == "active")).scalar_one()

    def _member_row(self, conn, p, user_id):
        m = conn.execute(select(memberships).where(memberships.c.user_id == user_id, memberships.c.org_id == p.org_id)).mappings().first()
        if m is None:
            raise NotFound("member not found")
        return m

    def change_role(self, p: Principal, user_id: str, role: Role) -> None:
        require(p, "members.manage")
        if user_id == p.user_id:
            raise Forbidden("you cannot change your own role")
        with self.engine.begin() as conn:
            m = self._member_row(conn, p, user_id)
            if m["role"] == Role.ADMIN.value and Role(role) != Role.ADMIN and self._admins(conn, p.org_id) <= 1:
                raise Forbidden("the last Platform Administrator cannot be demoted")
            conn.execute(update(memberships).where(memberships.c.user_id == user_id, memberships.c.org_id == p.org_id)
                         .values(role=Role(role).value))
            audit.append(conn, p.org_id, p.actor, "member.role_changed", f"user:{user_id}", {"from": m["role"], "to": Role(role).value})

    def set_user_status(self, p: Principal, user_id: str, status: str) -> None:
        require(p, "members.manage")
        if status not in ("active", "disabled"):
            raise ValueError("status must be active or disabled")
        if user_id == p.user_id:
            raise Forbidden("you cannot disable yourself")
        with self.engine.begin() as conn:
            m = self._member_row(conn, p, user_id)
            if status == "disabled" and m["role"] == Role.ADMIN.value and self._admins(conn, p.org_id) <= 1:
                raise Forbidden("the last Platform Administrator cannot be disabled")
            # per-organization: disabling someone here never affects their access to another organization
            conn.execute(update(memberships).where(memberships.c.user_id == user_id, memberships.c.org_id == p.org_id)
                         .values(status=status))
            audit.append(conn, p.org_id, p.actor, f"member.{status}", f"user:{user_id}", {})

    # ------------------------------------------------------------------ applications and policies
    def list_apps(self, p: Principal, purpose: str = "view") -> list[dict]:
        with self.engine.connect() as conn:
            return apps_visible(conn, p, purpose)

    def get_app(self, p: Principal, app_id: str, purpose: str = "view") -> dict:
        with self.engine.connect() as conn:
            return app_for(conn, p, app_id, purpose)

    def create_app(self, p: Principal, name: str, purpose: str, profile: str | None = None) -> str:
        require(p, "app.create")
        name = (name or "").strip()
        if not name or len(name) > 120:
            raise ValueError("name is required (max 120 characters)")
        if purpose not in policies.PURPOSE_TEMPLATES:
            raise ValueError("unknown purpose")
        with self.engine.begin() as conn:
            org = conn.execute(select(organizations).where(organizations.c.id == p.org_id)).mappings().first()
            profile = profile or org["industry"]
            if profile not in policies.PROFILES:
                raise ValueError("unknown profile")
            sp, topic = policies.PURPOSE_TEMPLATES[purpose]
            app_id = new_id()
            try:
                conn.execute(insert(applications).values(
                    id=app_id, org_id=p.org_id, name=name, purpose=purpose, owner_user_id=p.user_id, system_prompt=sp,
                    task_topic=topic, profile=profile, status="active", created_at=now(), updated_at=now(),
                    **policies.defaults(profile)))
            except IntegrityError:
                raise ValueError("an application with this name already exists")
            conn.execute(insert(app_members).values(org_id=p.org_id, app_id=app_id, user_id=p.user_id))
            audit.append(conn, p.org_id, p.actor, "app.created", f"app:{app_id}", {"name": name, "purpose": purpose, "profile": profile})
        return app_id

    def update_policy(self, p: Principal, app_id: str, values: dict) -> dict:
        with self.engine.begin() as conn:
            app = app_for(conn, p, app_id, "manage")
            clean = policies.validate(app["profile"], {**{k: app[k] for k in policies.EDITABLE}, **values})
            changed = {k: [app[k], v] for k, v in clean.items() if app[k] != v}
            if changed:
                conn.execute(update(applications).where(applications.c.id == app_id, applications.c.org_id == p.org_id)
                             .values(updated_at=now(), **clean))
                audit.append(conn, p.org_id, p.actor, "policy.updated", f"app:{app_id}", {"changes": changed})
            return changed

    def set_app_status(self, p: Principal, app_id: str, status: str) -> None:
        if status not in ("active", "paused"):
            raise ValueError("status must be active or paused")
        with self.engine.begin() as conn:
            app_for(conn, p, app_id, "manage")
            conn.execute(update(applications).where(applications.c.id == app_id, applications.c.org_id == p.org_id)
                         .values(status=status, updated_at=now()))
            audit.append(conn, p.org_id, p.actor, f"app.{status}", f"app:{app_id}", {})

    def app_assignees(self, p: Principal, app_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            app = app_for(conn, p, app_id, "view")
            if not (has(p.role, "app.view.org") or has(p.role, "app.manage.any") or app["owner_user_id"] == p.user_id):
                raise Forbidden("only the owner, administrators, analysts and auditors can list an application's members")
            q = (select(users.c.id, users.c.email, users.c.display_name).join(app_members, app_members.c.user_id == users.c.id)
                 .where(app_members.c.app_id == app_id, app_members.c.org_id == p.org_id).order_by(users.c.display_name))
            return [dict(r) for r in conn.execute(q).mappings()]

    def assign(self, p: Principal, app_id: str, user_id: str, add: bool = True) -> None:
        with self.engine.begin() as conn:
            app_for(conn, p, app_id, "manage")
            self._member_row(conn, p, user_id)          # must belong to the same org
            if add:
                if not conn.execute(select(app_members).where(app_members.c.app_id == app_id, app_members.c.user_id == user_id)).first():
                    conn.execute(insert(app_members).values(org_id=p.org_id, app_id=app_id, user_id=user_id))
            else:
                conn.execute(delete(app_members).where(app_members.c.app_id == app_id, app_members.c.user_id == user_id,
                                                       app_members.c.org_id == p.org_id))
            audit.append(conn, p.org_id, p.actor, "app.assigned" if add else "app.unassigned", f"app:{app_id}", {"user_id": user_id})

    def org_users(self, p: Principal) -> list[dict]:
        """Names of org members, for assigning applications (managers only)."""
        if not (has(p.role, "app.manage.any") or has(p.role, "app.manage.own")):
            raise Forbidden("not allowed")
        with self.engine.connect() as conn:
            q = (select(users.c.id, users.c.email, users.c.display_name, memberships.c.role)
                 .join(memberships, memberships.c.user_id == users.c.id).where(memberships.c.org_id == p.org_id, users.c.status == "active")
                 .order_by(users.c.display_name))
            return [dict(r) for r in conn.execute(q).mappings()]

    # ------------------------------------------------------------------ API keys (in-process SDK integration)
    def create_api_key(self, p: Principal, app_id: str) -> str:
        with self.engine.begin() as conn:
            app_for(conn, p, app_id, "manage")
            key = "ppag_" + secrets.token_urlsafe(32)
            kid = new_id()
            conn.execute(insert(api_keys).values(id=kid, org_id=p.org_id, app_id=app_id, prefix=key[:12], key_hash=hash_key(key),
                                                 created_by=p.user_id, created_at=now()))
            audit.append(conn, p.org_id, p.actor, "apikey.created", f"app:{app_id}", {"prefix": key[:12]})
        return key  # shown once; only the hash is stored

    def list_api_keys(self, p: Principal, app_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            app_for(conn, p, app_id, "manage")
            q = select(api_keys.c.id, api_keys.c.prefix, api_keys.c.created_at, api_keys.c.revoked_at, api_keys.c.last_used_at) \
                .where(api_keys.c.app_id == app_id, api_keys.c.org_id == p.org_id).order_by(api_keys.c.created_at.desc())
            return [dict(r) for r in conn.execute(q).mappings()]

    def revoke_api_key(self, p: Principal, key_id: str) -> None:
        with self.engine.begin() as conn:
            k = conn.execute(select(api_keys).where(api_keys.c.id == key_id, api_keys.c.org_id == p.org_id)).mappings().first()
            if k is None:
                raise NotFound("key not found")
            app_for(conn, p, k["app_id"], "manage")
            conn.execute(update(api_keys).where(api_keys.c.id == key_id).values(revoked_at=now()))
            audit.append(conn, p.org_id, p.actor, "apikey.revoked", f"app:{k['app_id']}", {"prefix": k["prefix"]})

    # ------------------------------------------------------------------ scans
    def purge_expired(self) -> int:
        with self.engine.begin() as conn:
            return conn.execute(delete(retained_inputs).where(retained_inputs.c.expires_at <= now())).rowcount

    def list_scans(self, p: Principal, decision: str | None = None, app_id: str | None = None, limit: int = 200) -> list[dict]:
        with self.engine.connect() as conn:
            q = select(scans, applications.c.name.label("app_name"), users.c.email.label("user_email")) \
                .join(applications, and_(applications.c.id == scans.c.app_id, applications.c.org_id == scans.c.org_id)) \
                .outerjoin(users, users.c.id == scans.c.user_id).where(scan_scope(conn, p))
            if decision:
                q = q.where(scans.c.decision == decision)
            if app_id:
                q = q.where(scans.c.app_id == app_id)
            rows = conn.execute(q.order_by(scans.c.created_at.desc()).limit(limit)).mappings().all()
            return [dict(r) for r in rows]

    def get_scan(self, p: Principal, scan_id: str) -> dict:
        self.purge_expired()
        with self.engine.connect() as conn:
            r = conn.execute(select(scans).where(scans.c.id == scan_id, scan_scope(conn, p))).mappings().first()
            if r is None:
                raise NotFound("scan not found")
            r = dict(r)
            r["reasons"] = json.loads(r.pop("reasons_json"))
            r["policy"] = json.loads(r.pop("policy_json"))
            r["retained_text"] = None
            if has(p.role, "scan.view.retained_text") or r["user_id"] == p.user_id:
                t = conn.execute(select(retained_inputs.c.redacted_text).where(retained_inputs.c.scan_id == scan_id,
                                                                               retained_inputs.c.org_id == p.org_id,
                                                                               retained_inputs.c.expires_at > now())).first()
                r["retained_text"] = t[0] if t else None
            return r

    def stats(self, p: Principal) -> dict:
        with self.engine.connect() as conn:
            scope = scan_scope(conn, p)
            since = now() - timedelta(hours=24)
            rows = conn.execute(select(scans.c.decision, func.count()).where(scope, scans.c.created_at >= since)
                                .group_by(scans.c.decision)).all()
            by = {d: n for d, n in rows}
            lat = [x[0] for x in conn.execute(select(scans.c.latency_ms).where(scope, scans.c.created_at >= since)).all()]
            open_reviews = 0
            if has(p.role, "review.view"):
                open_reviews = conn.execute(select(func.count()).select_from(reviews).where(reviews.c.org_id == p.org_id,
                                                                                         reviews.c.status == "open")).scalar_one()
            lat.sort()
            return {"total": sum(by.values()), "ALLOW": by.get("ALLOW", 0), "WARN": by.get("WARN", 0), "BLOCK": by.get("BLOCK", 0),
                    "open_reviews": open_reviews, "p50_ms": lat[len(lat) // 2] if lat else None,
                    "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None}

    # ------------------------------------------------------------------ WARN review workflow
    def list_reviews(self, p: Principal, status: str = "open") -> list[dict]:
        require(p, "review.view")
        with self.engine.connect() as conn:
            q = select(reviews, applications.c.name.label("app_name"), users.c.email.label("requester_email"), scans.c.score,
                       scans.c.reasons_json).join(applications, applications.c.id == reviews.c.app_id) \
                .join(scans, scans.c.id == reviews.c.scan_id).outerjoin(users, users.c.id == reviews.c.requester_id) \
                .where(reviews.c.org_id == p.org_id)
            if status != "all":
                q = q.where(reviews.c.status == status)
            return [dict(r) for r in conn.execute(q.order_by(reviews.c.created_at)).mappings()]

    def decide_review(self, p: Principal, review_id: str, verdict: str, note: str = "") -> None:
        """verdict: confirmed_injection | false_positive | released. A release is single-use, bound to the requester and
        the exact input (SHA-256), expires after 24 h, and still requires a full re-scan; it can never override BLOCK."""
        require(p, "review.decide")
        if verdict not in ("confirmed_injection", "false_positive", "released"):
            raise ValueError("unknown verdict")
        with self.engine.begin() as conn:
            r = conn.execute(select(reviews).where(reviews.c.id == review_id, reviews.c.org_id == p.org_id)).mappings().first()
            if r is None:
                raise NotFound("review not found")
            if r["status"] != "open":
                raise Forbidden("this review is already decided")
            if r["requester_id"] == p.user_id:
                raise Forbidden("you cannot review your own request")
            vals = dict(status=verdict, reviewer_id=p.user_id, note=(note or "")[:500], decided_at=now())
            if verdict == "released":
                vals["release_expires_at"] = now() + timedelta(hours=24)
            conn.execute(update(reviews).where(reviews.c.id == review_id, reviews.c.status == "open").values(**vals))
            audit.append(conn, p.org_id, p.actor, f"review.{verdict}", f"review:{review_id}", {"scan_id": r["scan_id"]})

    # ------------------------------------------------------------------ audit
    def list_audit(self, p: Principal, limit: int = 300) -> list[dict]:
        require(p, "audit.view")
        with self.engine.connect() as conn:
            q = select(audit_events).where(audit_events.c.org_id == p.org_id).order_by(audit_events.c.seq.desc()).limit(limit)
            return [dict(r) for r in conn.execute(q).mappings()]

    def verify_audit(self, p: Principal) -> dict:
        require(p, "audit.verify")
        with self.engine.begin() as conn:
            res = audit.verify(conn, p.org_id)
            audit.append(conn, p.org_id, p.actor, "audit.verified", f"org:{p.org_id}", {"ok": res["ok"], "checked": res["checked"]})
        return res

    def org(self, p: Principal) -> dict:
        with self.engine.connect() as conn:
            return dict(conn.execute(select(organizations).where(organizations.c.id == p.org_id)).mappings().one())

    def record_login(self, p: Principal, method: str) -> None:
        with self.engine.begin() as conn:
            audit.append(conn, p.org_id, p.actor, "auth.login", f"user:{p.user_id}", {"method": method})


def seed_demo(engine: Engine) -> None:
    """Synthetic demo data (only when demo mode is explicitly enabled). Idempotent."""
    with engine.connect() as conn:
        if conn.execute(select(func.count()).select_from(organizations)).scalar_one():
            return
    s = Services(engine)
    acme = s.create_organization("Acme Software (demo)", "software_it", "admin@acme.example", "Priya Admin")
    clinic = s.create_organization("City Clinic (demo)", "healthcare", "admin@clinic.example", "Dr. Rao Admin")
    from .authz import load_principal
    with engine.connect() as conn:
        pa = load_principal(conn, "admin@acme.example", acme)
        pc = load_principal(conn, "admin@clinic.example", clinic)
    ids = {}
    for email, name, role in [("analyst@acme.example", "Ananya Analyst", Role.ANALYST), ("dev@acme.example", "Karthik Developer", Role.DEVELOPER),
                              ("auditor@acme.example", "Sara Auditor", Role.AUDITOR), ("member@acme.example", "Ravi Member", Role.MEMBER)]:
        ids[email] = s.invite_member(pa, email, name, role)
    with engine.connect() as conn:
        pd = load_principal(conn, "dev@acme.example", acme)
    code = s.create_app(pd, "code-review-bot", "coding")
    mail = s.create_app(pa, "mail-assistant", "email", "general")
    docs = s.create_app(pa, "docs-assistant", "documents", "general")
    for a in (code, mail, docs):
        s.assign(pa, a, ids["member@acme.example"])
    s.assign(pa, mail, ids["dev@acme.example"])
    nurse = s.invite_member(pc, "member@clinic.example", "Nurse Member", Role.MEMBER)
    summ = s.create_app(pc, "discharge-summarizer", "documents")
    s.assign(pc, summ, nurse)

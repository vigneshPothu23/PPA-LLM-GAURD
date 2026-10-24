"""Who is calling (Rule 1). Three modes, chosen from configuration, never from the request:

  oidc          Streamlit's built-in OpenID Connect sign-in (st.login) with Google or Microsoft Entra ID, configured in
                secrets under [auth]. The verified e-mail address is looked up in the database on every rerun;
                an address that has not been invited gets no access.
  demo          Only when explicitly enabled (PPA_DEMO_MODE=1 or [ppa] demo_mode = true). A persona picker over
                synthetic seeded users, on a separate demo database. This is NOT authentication and the UI says so.
  unconfigured  Neither of the above: the app refuses to show any data (fail closed).

There is deliberately no password login: storing and checking passwords safely is not something to improvise.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from .db import organizations


def _get(m, key, default=None):
    try:
        return m.get(key, default) if m is not None else default
    except Exception:  # st.secrets raises when no secrets file exists
        return default


def oidc_configured(secrets: Mapping | None) -> bool:
    auth = _get(secrets, "auth")
    if not auth:
        return False
    if _get(auth, "client_id") and _get(auth, "server_metadata_url"):
        return True
    return any(isinstance(v, Mapping) and _get(v, "client_id") for v in _values(auth))


def _values(m):
    try:
        return list(m.values())
    except Exception:
        return []


def demo_enabled(secrets: Mapping | None, environ: Mapping | None = None) -> bool:
    environ = os.environ if environ is None else environ
    if str(environ.get("PPA_DEMO_MODE", "")).strip().lower() in ("1", "true", "yes"):
        return True
    return str(_get(_get(secrets, "ppa", {}), "demo_mode", "")).strip().lower() in ("1", "true", "yes")


def auth_mode(secrets: Mapping | None, environ: Mapping | None = None) -> str:
    if demo_enabled(secrets, environ):
        return "demo"
    if oidc_configured(secrets):
        return "oidc"
    return "unconfigured"


def oidc_providers(secrets: Mapping | None) -> list[str]:
    """[] -> single default provider (st.login()); otherwise named providers ([auth.google], [auth.microsoft])."""
    auth = _get(secrets, "auth") or {}
    try:
        return [k for k, v in auth.items() if isinstance(v, Mapping) and _get(v, "client_id")]
    except Exception:
        return []


MICROSOFT_ISSUER = "https://login.microsoftonline.com/"


def allowed_microsoft_tenants(secrets: Mapping | None) -> tuple[str, ...]:
    t = _get(_get(secrets, "ppa", {}) or {}, "microsoft_tenant_ids", ()) or ()
    return tuple(str(x).strip().lower() for x in ([t] if isinstance(t, str) else t) if str(x).strip())


def verified_email(user_info: Mapping, microsoft_tenants: tuple[str, ...] = ()) -> str | None:
    """E-mail from the identity provider's claims, or None if it cannot be trusted.

    Google: refused when email_verified is false.
    Microsoft Entra ID: the email claim is not a verified attribute in general (any tenant's administrator can set it),
    so a Microsoft identity is accepted only from a tenant listed in [ppa] microsoft_tenant_ids.
    """
    if not user_info or not _get(user_info, "is_logged_in", False):
        return None
    if _get(user_info, "email_verified", True) is False:
        return None
    if str(_get(user_info, "iss", "")).startswith(MICROSOFT_ISSUER):
        if str(_get(user_info, "tid", "")).lower() not in microsoft_tenants:
            return None
    email = _get(user_info, "email") or _get(user_info, "preferred_username")
    return email.strip().lower() if isinstance(email, str) and "@" in email else None


def bootstrap(engine: Engine, services, secrets: Mapping | None) -> str | None:
    """First deployment only: if the database has no organization yet, create one with the configured first
    Platform Administrator ([ppa] bootstrap_admin_email / bootstrap_org_name / bootstrap_industry). Idempotent."""
    ppa = _get(secrets, "ppa", {}) or {}
    email = _get(ppa, "bootstrap_admin_email")
    if not email:
        return None
    with engine.connect() as c:
        if c.execute(select(func.count()).select_from(organizations)).scalar_one() > 0:
            return None
    return services.create_organization(_get(ppa, "bootstrap_org_name", "My organization"),
                                        _get(ppa, "bootstrap_industry", "general"), email, _get(ppa, "bootstrap_admin_name", email))

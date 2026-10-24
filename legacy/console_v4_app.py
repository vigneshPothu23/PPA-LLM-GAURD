"""[LEGACY v4] PPA Guard - prompt-injection defence console for an organization's LLM applications.

Request path (enforced on the server for every scan, see src/ppa_guard/platform/gateway.py):
  sign-in -> role + organization + application scope -> Engine 2 (pretrained ProtectAI DeBERTa v2, ONNX, CPU)
  -> policy (ALLOW / WARN / BLOCK) -> Engine 1 (Polymorphic Prompt Assembling) -> audit record -> protected prompt.
No LLM is called and no model is trained.

Run locally:      PPA_DEMO_MODE=1 streamlit run streamlit_app.py
Production:       configure [auth] (Google or Microsoft OIDC) and PPA_DATABASE_URL (e.g. Neon free Postgres).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # v4 console, moved to legacy/ when the v5 app took streamlit_app.py
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app_pages import (applications, audit_log, command_centre, history, members, review_queue, scan_console,  # noqa: E402
                       system)
from app_pages.ui import Ctx, apply_theme  # noqa: E402
from ppa_guard.platform import db as dbm  # noqa: E402
from ppa_guard.platform import identity  # noqa: E402
from ppa_guard.platform.authz import AuthError, load_principal, memberships_for  # noqa: E402
from ppa_guard.platform.gateway import Gateway  # noqa: E402
from ppa_guard.platform.roles import ROLE_LABELS, has  # noqa: E402
from ppa_guard.platform.runtime import load_detector  # noqa: E402
from ppa_guard.platform.services import Services, seed_demo  # noqa: E402

st.set_page_config(page_title="PPA Guard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
apply_theme()
st.logo(str(ROOT / "assets/logo.svg"), size="large")


def _secrets():
    try:
        st.secrets.to_dict()  # raises when no secrets file exists
        return st.secrets
    except Exception:
        return {}


def _setting(name: str, secrets) -> str | None:
    """Environment first, then a top-level secret of the same name (Streamlit Cloud exposes both)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        return secrets.get(name)
    except Exception:
        return None


SECRETS = _secrets()
MODE = identity.auth_mode(SECRETS)


@st.cache_resource(show_spinner=False)
def get_engine(url: str | None, demo: bool, demo_db: str):
    # Demo mode always uses its own local database, so synthetic personas can never touch production data.
    eng = dbm.make_engine(f"sqlite:///{demo_db}" if demo else url)
    if demo:
        seed_demo(eng)
    return eng


@st.cache_resource(show_spinner="Loading Engine 2 (pretrained DeBERTa v2, ~740 MB on first start)...")
def get_detector(model_dir: str | None):
    return load_detector(model_dir)


if MODE == "unconfigured":
    st.error("Sign-in is not configured, so PPA Guard will not show any data (fail closed).")
    st.markdown("Configure Google or Microsoft sign-in under `[auth]` in the app secrets, or set `PPA_DEMO_MODE=1` "
                "to explore with synthetic demo users. See `docs/DEPLOY.md`.")
    st.stop()

engine = get_engine(_setting("PPA_DATABASE_URL", SECRETS), MODE == "demo", os.environ.get("PPA_DEMO_DB", str(ROOT / "ppa_guard_demo.db")))
services = Services(engine)
detector, model_status = get_detector(_setting("PPA_MODEL_DIR", SECRETS))
gateway = Gateway(engine, detector, model_status.error or "Engine 2 not loaded")

# ---------------------------------------------------------------- identity (Rule 1), re-checked on every rerun
email = None
if MODE == "oidc":
    identity.bootstrap(engine, services, SECRETS)
    if not st.user.is_logged_in:
        st.markdown("### Sign in")
        st.caption("Access is by invitation. Use the work account your administrator invited.")
        providers = identity.oidc_providers(SECRETS)
        for prov in providers or [None]:
            label = {"google": "Continue with Google", "microsoft": "Continue with Microsoft"}.get(prov, "Sign in")
            if st.button(label, type="primary", key=f"login-{prov}"):
                st.login(prov) if prov else st.login()
        st.stop()
    email = identity.verified_email(st.user, identity.allowed_microsoft_tenants(SECRETS))
    if email is None:
        st.error("Your identity provider did not supply a verified e-mail address this deployment accepts.")
        st.caption("Google accounts need a verified address; Microsoft accounts must belong to an allowed tenant.")
        if st.button("Sign out"):
            st.logout()
        st.stop()
else:  # demo
    with engine.connect() as c:
        from sqlalchemy import select
        people = c.execute(select(dbm.users.c.email, dbm.users.c.display_name).order_by(dbm.users.c.email)).all()
    with st.sidebar:
        choice = st.selectbox("Demo persona", [e for e, _ in people], key="persona",
                              format_func=lambda e: f"{dict(people)[e]} - {e}")
    email = choice

try:
    with engine.connect() as c:
        orgs = memberships_for(c, email) if email else []
        org_ids = [o["org_id"] for o in orgs]
        if len(orgs) > 1:
            with st.sidebar:
                st.selectbox("Organization", org_ids, key="org_id", format_func=lambda i: next(o["name"] for o in orgs if o["org_id"] == i))
        principal = load_principal(c, email, st.session_state.get("org_id") if st.session_state.get("org_id") in org_ids else None)
except AuthError as e:
    st.error(f"No access: {e}.")
    st.caption("Ask a Platform Administrator in your organization to invite this e-mail address.")
    if MODE == "oidc" and st.button("Sign out"):
        st.logout()
    st.stop()

if MODE == "oidc" and st.session_state.get("_login_recorded") != (email, principal.org_id):
    services.record_login(principal, "oidc")
    st.session_state["_login_recorded"] = (email, principal.org_id)

ctx = Ctx(services, gateway, principal, services.org(principal), MODE, model_status, engine.dialect.name,
          engine.dialect.name != "sqlite")


def guarded(fn):
    def run():
        if MODE == "demo":
            st.markdown('<div class="demo-banner"><b>DEMO MODE</b> - synthetic users and data on a separate local database. '
                        'Choosing a persona is not authentication; production uses Google or Microsoft sign-in.</div>',
                        unsafe_allow_html=True)
        try:
            fn(ctx)
        except AuthError as e:  # the server refused; the UI only reports it
            st.error(f"Not permitted: {e}")
    run.__name__ = fn.__module__.rsplit(".", 1)[-1]
    return run


r = principal.role
pages = [st.Page(guarded(command_centre.render), title="Command Centre", icon=":material/space_dashboard:",
                 url_path="command-centre", default=True)]
if has(r, "scan.run.any_app") or has(r, "scan.run.assigned_app"):
    pages.append(st.Page(guarded(scan_console.render), title="Scan Console", icon=":material/document_scanner:", url_path="scan"))
if has(r, "review.view"):
    pages.append(st.Page(guarded(review_queue.render), title="Review Queue", icon=":material/rule:", url_path="reviews"))
pages.append(st.Page(guarded(history.render), title="Scan History", icon=":material/history:", url_path="history"))
if has(r, "app.view.org") or has(r, "app.create"):
    pages.append(st.Page(guarded(applications.render), title="Applications & Policy", icon=":material/apps:", url_path="applications"))
if has(r, "members.view"):
    pages.append(st.Page(guarded(members.render), title="Members", icon=":material/group:", url_path="members"))
if has(r, "audit.view"):
    pages.append(st.Page(guarded(audit_log.render), title="Audit Log", icon=":material/verified_user:", url_path="audit"))
pages.append(st.Page(guarded(system.render), title="System & Integrations", icon=":material/hub:", url_path="system"))

nav = st.navigation(pages)
with st.sidebar:
    st.markdown(f"**{principal.label}**  \n{ROLE_LABELS[r]}  \n<span style='color:#9AA6AB'>{ctx.org['name']}</span>",
                unsafe_allow_html=True)
    if not model_status.loaded:
        st.error("Engine 2 not loaded - every scan is BLOCKED (fail closed).")
    if MODE == "oidc" and st.button("Sign out", use_container_width=True):
        st.logout()
nav.run()

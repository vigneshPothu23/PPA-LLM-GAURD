"""Streamlit entry logic (called from streamlit_app.py): configuration, identity, role-aware navigation."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from sqlalchemy import select

from ..core import db as dbm
from ..core import identity
from ..core.authz import AuthError, load_principal, memberships_for
from ..core.roles import ROLE_LABELS, has
from ..core.runtime import load_detector
from ..core.services import Services, seed_demo
from ..core.workflow import SecurityWorkflow
from .pages import (applications, audit_log, command_centre, history, members, model, review_queue, scan_console,
                    system)
from .theme import Ctx, apply_theme

SETTINGS = ("PPAGUARD_DATABASE_URL", "PPAGUARD_MODEL_DIR", "PPAGUARD_MODEL", "PPAGUARD_THREADS", "PPAGUARD_CACHE")


def _secrets():
    try:
        st.secrets.to_dict()          # raises when no secrets file exists
        return st.secrets
    except Exception:
        return {}


def _settings(secrets) -> dict:
    """Environment first, then a top-level secret of the same name."""
    out = {}
    for k in SETTINGS:
        v = os.environ.get(k)
        if not v:
            try:
                v = secrets.get(k)
            except Exception:
                v = None
        if v:
            out[k] = str(v)
    return out


@st.cache_resource(show_spinner=False)
def _engine(url: str | None, demo: bool, demo_db: str):
    # Demo mode always uses its own local database, so synthetic personas can never touch production data.
    eng = dbm.make_engine(f"sqlite:///{demo_db}" if demo else url)
    if demo:
        seed_demo(eng)
    return eng


@st.cache_resource(show_spinner="Loading Engine 2...")
def _detector(model_dir: str | None, model: str | None, threads: str | None, cache: str | None):
    env = {k: v for k, v in (("PPAGUARD_MODEL_DIR", model_dir), ("PPAGUARD_MODEL", model), ("PPAGUARD_THREADS", threads),
                             ("PPAGUARD_CACHE", cache)) if v}
    return load_detector(env)


def main(root: Path) -> None:
    st.set_page_config(page_title="PPA Guard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    st.logo(str(root / "assets/logo.svg"), size="large")
    secrets = _secrets()
    mode = identity.auth_mode(secrets)
    if mode == "unconfigured":
        st.error("Sign-in is not configured, so PPA Guard will not show any data (fail closed).")
        st.markdown("Configure Google or Microsoft sign-in under `[auth]` in the app secrets, or set "
                    "`PPAGUARD_DEMO_MODE=1` (or `[ppaguard] demo_mode = true`) to explore with synthetic demo users. "
                    "See `docs/v5/DEPLOY.md`.")
        st.stop()

    cfg = _settings(secrets)
    engine = _engine(cfg.get("PPAGUARD_DATABASE_URL"), mode == "demo",
                     os.environ.get("PPAGUARD_DEMO_DB", str(root / "ppaguard_demo.db")))
    services = Services(engine)
    detector, model_status = _detector(cfg.get("PPAGUARD_MODEL_DIR"), cfg.get("PPAGUARD_MODEL"), cfg.get("PPAGUARD_THREADS"),
                                       cfg.get("PPAGUARD_CACHE"))
    workflow = SecurityWorkflow(engine, detector, model_status.error)

    # ------------------------------------------------------------ identity (Rule 1), re-checked on every rerun
    if mode == "oidc":
        identity.bootstrap(engine, services, secrets)
        if not st.user.is_logged_in:
            st.markdown("### Sign in")
            st.caption("Access is by invitation. Use the work account your administrator invited.")
            for prov in identity.oidc_providers(secrets) or [None]:
                label = {"google": "Continue with Google", "microsoft": "Continue with Microsoft"}.get(prov, "Sign in")
                if st.button(label, type="primary", key=f"login-{prov}"):
                    st.login(prov) if prov else st.login()
            st.stop()
        email = identity.verified_email(st.user, identity.allowed_microsoft_tenants(secrets))
        if email is None:
            st.error("Your identity provider did not supply a verified e-mail address this deployment accepts.")
            st.caption("Google accounts need a verified address; Microsoft accounts must belong to an allowed tenant.")
            if st.button("Sign out"):
                st.logout()
            st.stop()
    else:  # demo
        with engine.connect() as c:
            people = c.execute(select(dbm.users.c.email, dbm.users.c.display_name).order_by(dbm.users.c.email)).all()
        names = dict(people)
        with st.sidebar:
            email = st.selectbox("Demo persona", [e for e, _ in people], key="persona",
                                 format_func=lambda e: f"{names[e]} - {e}")

    try:
        with engine.connect() as c:
            orgs = memberships_for(c, email) if email else []
            org_ids = [o["org_id"] for o in orgs]
            if len(orgs) > 1:
                with st.sidebar:
                    st.selectbox("Organization", org_ids, key="org_id",
                                 format_func=lambda i: next(o["name"] for o in orgs if o["org_id"] == i))
            chosen = st.session_state.get("org_id")
            principal = load_principal(c, email, chosen if chosen in org_ids else None)
    except AuthError as e:
        st.error(f"No access: {e}.")
        st.caption("Ask a Platform Administrator in your organization to invite this e-mail address.")
        if mode == "oidc" and st.button("Sign out"):
            st.logout()
        st.stop()

    if mode == "oidc" and st.session_state.get("_login_recorded") != (email, principal.org_id):
        services.record_login(principal, "oidc")
        st.session_state["_login_recorded"] = (email, principal.org_id)

    ctx = Ctx(services, workflow, principal, services.org(principal), mode, model_status, engine.dialect.name,
              engine.dialect.name != "sqlite")

    def guarded(fn):
        def run():
            if mode == "demo":
                st.markdown('<div class="demo-banner"><b>DEMO MODE</b> - synthetic users and data on a separate local '
                            'database. Choosing a persona is not authentication; production uses Google or Microsoft '
                            'sign-in.</div>', unsafe_allow_html=True)
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
        pages.append(st.Page(guarded(applications.render), title="Applications & Policy", icon=":material/apps:",
                             url_path="applications"))
    if has(r, "model.view"):
        pages.append(st.Page(guarded(model.render), title="Engine 2 Model", icon=":material/model_training:", url_path="model"))
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
        elif model_status.provenance == "test-only":
            st.warning("Engine 2 is a TEST ARTIFACT (random weights).")
        elif model_status.detail.get("status") != "validated":
            st.warning("Engine 2 is EXPERIMENTAL - not validated for production use. See Engine 2 Model / System.")
        if mode == "oidc" and st.button("Sign out", width="stretch"):
            st.logout()
    nav.run()

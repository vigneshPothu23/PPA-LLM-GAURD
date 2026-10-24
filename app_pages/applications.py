"""Applications & Policy: register LLM applications, set their policy within profile bounds, assign members,
issue API keys for the in-process SDK integration. Every action is re-authorized in Services."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ppa_guard.platform import policies
from ppa_guard.platform.authz import AuthError
from ppa_guard.platform.roles import has

from .ui import AMBER, esc, header, pill, ts


def _can_manage(ctx, app_id) -> bool:
    try:
        ctx.s.get_app(ctx.p, app_id, "manage")
        return True
    except AuthError:
        return False


def render(ctx) -> None:
    header("Applications & Policy", "Each LLM application has its own policy, members and API keys.")
    apps = ctx.s.list_apps(ctx.p)
    if has(ctx.p.role, "app.create"):
        with st.expander("Register an application"):
            with st.form("new_app", clear_on_submit=True):
                name = st.text_input("Name", max_chars=120)
                purpose = st.selectbox("Purpose", list(policies.PURPOSE_TEMPLATES))
                profile = st.selectbox("Industry profile", list(policies.PROFILES), index=list(policies.PROFILES).index(ctx.org["industry"]),
                                       format_func=lambda k: policies.PROFILES[k]["label"])
                st.caption(policies.PROFILES[profile]["note"])
                if st.form_submit_button("Register", type="primary"):
                    try:
                        ctx.s.create_app(ctx.p, name, purpose, profile)
                        st.success("Registered. You are its owner.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    if not apps:
        st.caption("No applications visible to you.")
        return
    st.dataframe(pd.DataFrame([{"Application": a["name"], "Purpose": a["purpose"], "Profile": policies.PROFILES[a["profile"]]["label"],
                                "WARN >": a["warn_threshold"], "BLOCK >": a["block_threshold"], "WARN action": a["warn_action"],
                                "Retention (days)": a["retention_days"], "Status": a["status"]} for a in apps]),
                 hide_index=True, use_container_width=True)

    names = {a["id"]: a for a in apps}
    app_id = st.selectbox("Open application", list(names), format_func=lambda i: names[i]["name"], key="ap_pick")
    app = names[app_id]
    manage = _can_manage(ctx, app_id)
    prof = policies.PROFILES[app["profile"]]
    st.markdown(f"### {esc(app['name'])} " + ("" if policies.is_validated(app) else pill("thresholds not validated", AMBER)),
                unsafe_allow_html=True)
    st.caption(f"{prof['label']} profile: {prof['note']}")
    tabs = st.tabs(["Policy", "Members", "API keys", "Prompt template"])

    with tabs[0]:
        with st.form(f"pol-{app_id}"):
            c1, c2 = st.columns(2)
            warn = c1.slider("WARN if P(injection) >", 0.05, 0.99, float(app["warn_threshold"]), 0.01, disabled=not manage)
            block = c2.slider("BLOCK if P(injection) >", 0.05, 0.99, float(app["block_threshold"]), 0.01, disabled=not manage)
            st.caption("0.50 / 0.92 are the validated defaults (model argmax / ProtectAI LLM Guard). Any other value is "
                       "labelled not validated.")
            c3, c4, c5, c6 = st.columns(4)
            wa = c3.selectbox("WARN action", ["hold", "allow_with_flag"], index=["hold", "allow_with_flag"].index(app["warn_action"]),
                              disabled=not manage, help="hold = analyst review; allow_with_flag = forward and flag")
            fa = c4.selectbox("Separator forgery", ["WARN", "BLOCK"], index=["WARN", "BLOCK"].index(app["forgery_action"]), disabled=not manage)
            ha = c5.selectbox("Hidden characters", ["WARN", "BLOCK"], index=["WARN", "BLOCK"].index(app["hidden_chars_action"]), disabled=not manage)
            ta = c6.selectbox("Over 512 tokens", ["WARN", "BLOCK"], index=["WARN", "BLOCK"].index(app["truncated_action"]), disabled=not manage)
            c7, c8, c9 = st.columns(3)
            mc = c7.number_input("Max characters", 1, 20000, int(app["max_chars"]), disabled=not manage)
            rd = c8.number_input(f"Retention days (max {prof['max_retention_days']})", 0, max(prof["max_retention_days"], 0),
                                 int(app["retention_days"]), disabled=not manage or prof["max_retention_days"] == 0)
            rl = c9.number_input("Scans per minute per user", 1, 600, int(app["rate_limit_per_min"]), disabled=not manage)
            if st.form_submit_button("Save policy", type="primary", disabled=not manage):
                try:
                    ch = ctx.s.update_policy(ctx.p, app_id, dict(warn_threshold=warn, block_threshold=block, warn_action=wa,
                                                                 forgery_action=fa, hidden_chars_action=ha, truncated_action=ta,
                                                                 max_chars=mc, retention_days=rd, rate_limit_per_min=rl))
                    st.success(f"Saved {len(ch)} change(s); recorded in the audit log." if ch else "No changes.")
                except ValueError as e:
                    st.error(str(e))
        if manage:
            new = "paused" if app["status"] == "active" else "active"
            if st.button("Pause application" if new == "paused" else "Resume application", key=f"st-{app_id}"):
                ctx.s.set_app_status(ctx.p, app_id, new)
                st.rerun()
        else:
            st.caption("Read-only: you can view this application but not change it.")

    with tabs[1]:
        try:
            assigned = ctx.s.app_assignees(ctx.p, app_id)
        except AuthError as e:
            assigned = None
            st.caption(f"Not shown: {e}.")
        if assigned is not None:
            st.dataframe(pd.DataFrame(assigned or [{"display_name": "-", "email": "-"}])[["display_name", "email"]]
                         .rename(columns={"display_name": "Name", "email": "E-mail"}), hide_index=True, use_container_width=True)
        if manage:
            people = {u["id"]: u for u in ctx.s.org_users(ctx.p)}
            ids_in = {u["id"] for u in assigned or []}
            c1, c2 = st.columns(2)
            add = c1.selectbox("Assign member", [i for i in people if i not in ids_in] or [None],
                               format_func=lambda i: "-" if i is None else f"{people[i]['display_name']} ({people[i]['role']})", key=f"as-{app_id}")
            if c1.button("Assign", key=f"asb-{app_id}", disabled=add is None):
                ctx.s.assign(ctx.p, app_id, add)
                st.rerun()
            rem = c2.selectbox("Remove member", sorted(ids_in) or [None],
                               format_func=lambda i: "-" if i is None else people.get(i, {"display_name": i})["display_name"], key=f"rm-{app_id}")
            if c2.button("Remove", key=f"rmb-{app_id}", disabled=rem is None):
                ctx.s.assign(ctx.p, app_id, rem, add=False)
                st.rerun()

    with tabs[2]:
        if not manage:
            st.caption("API keys are visible to the application's owner and Platform Administrators only.")
        else:
            st.caption("A key lets one Python application call PPA Guard in-process for this application only "
                       "(see System & Integrations). The full key is shown once; only its hash is stored.")
            if st.button("Create API key", key=f"k-{app_id}"):
                st.session_state[f"newkey-{app_id}"] = ctx.s.create_api_key(ctx.p, app_id)
            if st.session_state.get(f"newkey-{app_id}"):
                st.warning("Copy this key now. It will not be shown again.")
                st.code(st.session_state.pop(f"newkey-{app_id}"), language="text")
            keys = ctx.s.list_api_keys(ctx.p, app_id)
            for k in keys:
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"`{k['prefix']}...` created {ts(k['created_at'])} · last used {ts(k['last_used_at']) or 'never'}"
                            + (f" · **revoked** {ts(k['revoked_at'])}" if k["revoked_at"] else ""))
                if not k["revoked_at"] and c2.button("Revoke", key=f"rv-{k['id']}"):
                    ctx.s.revoke_api_key(ctx.p, k["id"])
                    st.rerun()

    with tabs[3]:
        st.caption("Trusted instructions Engine 1 places around the untrusted input (set from the purpose template).")
        st.code(app["system_prompt"], language="text")
        st.markdown(f"Task topic: `{app['task_topic']}`")

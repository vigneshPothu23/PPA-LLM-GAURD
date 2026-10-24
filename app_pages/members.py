"""Members (Platform Administrator manages; Auditor views). Invitations are by e-mail; the person signs in with
Google or Microsoft using that address."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ppa_guard.platform.roles import ROLE_LABELS, Role, has

from .ui import header


def render(ctx) -> None:
    manage = has(ctx.p.role, "members.manage")
    header("Members", "People in this organization and their role." + ("" if manage else " Read-only."))
    rows = ctx.s.list_members(ctx.p)
    st.dataframe(pd.DataFrame([{"Name": r["display_name"], "E-mail": r["email"], "Role": ROLE_LABELS[Role(r["role"])],
                                "Status": r["status"]} for r in rows]), hide_index=True, use_container_width=True)
    if not manage:
        return
    roles = list(Role)
    with st.expander("Invite a member"):
        with st.form("invite", clear_on_submit=True):
            email = st.text_input("Work e-mail (must match their Google or Microsoft sign-in)")
            name = st.text_input("Display name")
            role = st.selectbox("Role", roles, index=roles.index(Role.MEMBER), format_func=lambda x: ROLE_LABELS[x])
            if st.form_submit_button("Invite", type="primary"):
                try:
                    ctx.s.invite_member(ctx.p, email, name, role)
                    st.success("Invited. They can sign in now.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    people = {r["id"]: r for r in rows if r["id"] != ctx.p.user_id}
    if not people:
        return
    st.markdown("#### Change a member")
    uid = st.selectbox("Member", list(people), format_func=lambda i: f"{people[i]['display_name']} ({people[i]['email']})", key="m_pick")
    who = people[uid]
    c1, c2 = st.columns(2)
    new_role = c1.selectbox("Role", roles, index=roles.index(Role(who["role"])), format_func=lambda x: ROLE_LABELS[x], key="m_role")
    if c1.button("Change role", disabled=new_role.value == who["role"]):
        ctx.s.change_role(ctx.p, uid, new_role)
        st.rerun()
    target = "disabled" if who["status"] == "active" else "active"
    c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if c2.button("Disable access" if target == "disabled" else "Re-enable access", use_container_width=True):
        ctx.s.set_user_status(ctx.p, uid, target)
        st.rerun()
    st.caption("You cannot change your own role, and the last Platform Administrator cannot be demoted or disabled. "
               "Disabling takes effect on the person's next action.")

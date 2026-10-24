"""Audit Log: hash-chained events for this organization, with on-demand chain verification."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ppa_guard.platform.roles import has

from .ui import GREEN, RED, header, pill, ts


def render(ctx) -> None:
    header("Audit Log", "Every sign-in, decision, review, policy and membership change. Each event's hash covers the "
                        "previous one, so edits, deletions and re-ordering are detectable.")
    if has(ctx.p.role, "audit.verify"):
        if st.button("Verify chain", type="primary"):
            st.session_state["audit_verify"] = ctx.s.verify_audit(ctx.p)
        res = st.session_state.get("audit_verify")
        if res:
            if res["ok"]:
                st.markdown(pill("CHAIN INTACT", GREEN) + f" &nbsp; {res['checked']} events checked", unsafe_allow_html=True)
            else:
                st.markdown(pill("CHAIN BROKEN", RED) + f" &nbsp; at event {res['broken_at']}: {res['reason']}", unsafe_allow_html=True)
    st.caption("Limitation: this detects tampering after the fact. Someone with direct write access to the database could "
               "rewrite the entire chain; export it regularly to keep an outside copy.")
    rows = ctx.s.list_audit(ctx.p, limit=1000)
    df = pd.DataFrame([{"#": r["seq"], "Time (UTC)": ts(r["created_at"]), "Actor": r["actor"], "Action": r["action"],
                        "Target": r["target"], "Details": r["details_json"], "Hash": r["hash"][:16] + "..."} for r in rows])
    actions = sorted({r["action"] for r in rows})
    pick = st.multiselect("Filter actions", actions, key="a_filter")
    if pick:
        df = df[df["Action"].isin(pick)]
    st.dataframe(df, hide_index=True, use_container_width=True, height=460)
    st.download_button("Export CSV", df.to_csv(index=False).encode(), "audit_log.csv", "text/csv")

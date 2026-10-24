"""Scan History: decision records within the caller's scope (organization, own applications, or own scans)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ppa_guard.platform.roles import has

from .ui import header, pill, ts


def render(ctx) -> None:
    scope = ("all scans in the organization" if has(ctx.p.role, "scan.view.org") else
             "scans in your applications and your own scans" if has(ctx.p.role, "scan.view.own_apps") else "your own scans")
    header("Scan History", f"Showing {scope}. Only a SHA-256 of each input is stored unless the application retains redacted text.")
    apps = {a["id"]: a["name"] for a in ctx.s.list_apps(ctx.p)}
    c1, c2 = st.columns(2)
    dec = c1.selectbox("Decision", ["All", "ALLOW", "WARN", "BLOCK"], key="h_dec")
    app = c2.selectbox("Application", ["All"] + list(apps), format_func=lambda i: "All" if i == "All" else apps[i], key="h_app")
    rows = ctx.s.list_scans(ctx.p, None if dec == "All" else dec, None if app == "All" else app, limit=500)
    if not rows:
        st.caption("No scans match.")
        return
    df = pd.DataFrame([{"Time (UTC)": ts(r["created_at"]), "Scan": r["id"][:8], "Application": r["app_name"],
                        "By": r["user_email"] or "API key", "Channel": r["channel"], "Score": r["score"],
                        "Decision": r["decision"], "Outcome": r["outcome"], "Input SHA-256": r["input_sha256"][:16] + "...",
                        "Chars": r["input_chars"]} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True, height=360)
    st.download_button("Download CSV (metadata only)", df.to_csv(index=False).encode(), "scan_history.csv", "text/csv")

    ids = {r["id"][:8]: r["id"] for r in rows}
    pick = st.selectbox("Open a scan record", list(ids), key="h_pick")
    d = ctx.s.get_scan(ctx.p, ids[pick])
    with st.container(border=True):
        st.markdown(f"{pill(d['decision'])} &nbsp; **{d['outcome']}** · score {d['score']} · band {d['detector_band']} · "
                    f"tokens {d['n_tokens']} · Engine 1 {'applied' if d['engine1_applied'] else 'not applied'} · model `{d['model_id']}`",
                    unsafe_allow_html=True)
        for reason in d["reasons"]:
            st.markdown(f"- {reason}")
        st.caption("Policy in force at decision time")
        st.json(d["policy"], expanded=False)
        if d["retained_text"]:
            st.caption("Retained input (redacted, deleted at expiry)")
            st.code(d["retained_text"], language="text")

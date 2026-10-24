"""Command Centre: the caller's scoped view of recent decisions and system status."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

import pandas as pd
import streamlit as st

from ppa_guard.platform.db import now
from ppa_guard.platform.roles import ROLE_LABELS, has

from .ui import AMBER, GREEN, RED, chip, header, kpis, ms, pill


def render(ctx) -> None:
    p = ctx.p
    scope = {"scan.view.org": "all applications in the organization", "scan.view.own_apps": "your applications",
             "scan.view.self": "your own scans"}
    what = next(v for k, v in scope.items() if has(p.role, k))
    header("Command Centre", f"Hello {p.label} ({ROLE_LABELS[p.role]}). Last 24 hours, {what}.")
    st_ = ctx.s.stats(p)
    kpis([("Scans · 24 h", f"{st_['total']:,}", "recorded decisions"),
          ("Blocked", f"{st_['BLOCK']:,}", pill(f"{st_['BLOCK'] / st_['total']:.0%}" if st_["total"] else "0%", RED)),
          ("WARN", f"{st_['WARN']:,}", pill(f"{st_['open_reviews']} open reviews", AMBER) if has(p.role, "review.view") else "held or flagged"),
          ("Latency p95", ms(st_["p95_ms"]), f"p50 {ms(st_['p50_ms'])}")])

    rows = ctx.s.list_scans(p, limit=1000)
    st.markdown("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Decisions · last 7 days**")
        since = now() - timedelta(days=7)
        recent = [r for r in rows if _aware(r["created_at"]) >= since]
        if recent:
            cnt = Counter((_aware(r["created_at"]).date().isoformat(), r["decision"]) for r in recent)
            days = sorted({d for d, _ in cnt})
            df = pd.DataFrame({k: [cnt.get((d, k), 0) for d in days] for k in ("ALLOW", "WARN", "BLOCK")}, index=days)
            st.bar_chart(df, color=[GREEN, AMBER, RED], height=260, x_label="Day (UTC)", y_label="Scans")
        else:
            st.caption("No scans in the last 7 days.")
    with c2:
        st.markdown("**Recent scans** · metadata only")
        if rows:
            st.dataframe(pd.DataFrame([{"Time": _aware(r["created_at"]).strftime("%m-%d %H:%M"), "App": r["app_name"],
                                        "Score": r["score"], "Decision": r["decision"]} for r in rows[:8]]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("No scans yet.")

    st.markdown("**System**")
    m = ctx.model
    st.markdown(chip(f"Engine 2: {m.model_id or 'not loaded'} ({m.verified})",
                     (True if m.source == "pinned-download" else None) if m.loaded else False)
                + chip("Engine 1: PPA, byte-identical to SDK b3c20a3, hardened mode", True)
                + chip(f"Database: {ctx.db_backend}" + ("" if ctx.db_persistent else " (not persistent on Streamlit Cloud)"),
                       True if ctx.db_persistent else None)
                + chip("LLM: not connected (placeholder)", None), unsafe_allow_html=True)


def _aware(d):
    from datetime import timezone
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d

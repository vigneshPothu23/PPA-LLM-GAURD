"""Review Queue (Security Analyst, Platform Administrator): decide WARN inputs that were held."""

from __future__ import annotations

import json

import streamlit as st

from ..theme import AMBER, esc, header, pill, ts


def render(ctx) -> None:
    header("Review Queue", "WARN inputs held for a decision. Releasing one lets the requester submit the same text once, "
                           "within 24 hours; it is fully re-scanned and a BLOCK can never be released.")
    status = st.segmented_control("Show", ["open", "released", "release_used", "confirmed_injection", "false_positive", "all"],
                                  default="open", key="rq_status")
    items = ctx.s.list_reviews(ctx.p, status or "open")
    if not items:
        st.success("Nothing waiting.") if (status or "open") == "open" else st.caption("No reviews with this status.")
        return
    can_decide = True
    for r in items:
        with st.container(border=True):
            a, b = st.columns([4, 2])
            a.markdown(f"**{esc(r['app_name'])}** · requested by {esc(r['requester_email'] or 'API key')} · {ts(r['created_at'])}  \n"
                       f"{pill('WARN', AMBER)} score {r['score'] if r['score'] is not None else 'n/a'} · review `{r['id'][:8]}`",
                       unsafe_allow_html=True)
            b.markdown(f"Status: **{r['status']}**")
            for reason in json.loads(r["reasons_json"]):
                st.markdown(f"- {reason}")
            detail = ctx.s.get_scan(ctx.p, r["scan_id"])
            if detail["retained_text"]:
                st.caption("Retained input (redacted)")
                st.code(detail["retained_text"], language="text")
            else:
                st.caption("Input text is not retained for this application (data minimisation); decide from the reasons and "
                           "score, or ask the requester.")
            if r["status"] == "open" and can_decide:
                note = st.text_input("Note (optional)", key=f"note-{r['id']}", max_chars=500)
                c1, c2, c3 = st.columns(3)
                for col, verdict_, label, kind in ((c1, "released", "Release once", "primary"),
                                                   (c2, "false_positive", "Mark false positive", "secondary"),
                                                   (c3, "confirmed_injection", "Confirm injection", "secondary")):
                    if col.button(label, key=f"{verdict_}-{r['id']}", type=kind, width="stretch"):
                        ctx.s.decide_review(ctx.p, r["id"], verdict_, note)
                        st.rerun()

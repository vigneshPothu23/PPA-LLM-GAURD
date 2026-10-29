"""System & Integrations: honest status of every component, the exact integration boundary, and the role matrix."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.roles import PERMISSIONS, ROLE_LABELS, Role
from ..theme import chip, header

SDK = '''# In your own Python application (same database as this console):
#   pip install -r requirements.txt     # from the PPA Guard repository; add its src/ folder to PYTHONPATH
import os
from ppaguard.sdk import PPAGuardClient

guard = PPAGuardClient(api_key=os.environ["PPAGUARD_API_KEY"])   # key from Applications & Policy -> API keys
r = guard.protect(untrusted_text)
if r.allowed:                                   # ALLOW (or a WARN the application forwards with a flag)
    reply = my_llm(r.protected_prompt)          # your LLM call - PPA Guard never calls one
    leaked = guard.check_response(reply, r)     # did the model echo this request's boundary marker?
else:                                           # BLOCK, or WARN held for analyst review
    refuse(r.decision, r.reasons)
'''


def render(ctx) -> None:
    header("System & Integrations", "What is running, what is connected, and what PPA Guard does not do.")
    m = ctx.model
    st.markdown(
        chip(f"Engine 2: {m.model_id or 'not loaded'}", (None if m.provenance == "test-only" else True) if m.loaded else False)
        + (chip(m.detail.get("provenance_text", ""), None) if m.loaded else "")
        + (chip("Engine 2 context signal: " + ("active" if getattr(ctx.wf.detector, "context_signal", None) else "off"),
                True if getattr(ctx.wf.detector, "context_signal", None) else None) if m.loaded else "")
        + chip("Engine 1: polymorphic prompt assembly", True)
        + chip(f"Database: {ctx.db_backend}" + (" (persistent)" if ctx.db_persistent else " (local file; resets on Streamlit Cloud restarts)"),
               True if ctx.db_persistent else None)
        + chip(f"Sign-in: {'Google / Microsoft OIDC' if ctx.mode == 'oidc' else 'DEMO personas (not authentication)'}",
               True if ctx.mode == "oidc" else None)
        + chip("LLM: not connected", None), unsafe_allow_html=True)
    if m.error:
        st.error(f"Engine 2 failed to load: {m.error}. Every scan is BLOCKED until a model loads (fail closed).")

    st.markdown("#### How a request is handled")
    st.markdown(
        "1. **Sign-in and role** - the caller's organization, role and application assignment are re-read from the database.\n"
        "2. **Engine 2** - the transformer detector scores the text (overlapping 512-token windows for long input) and "
        "returns a calibrated P(injection); its context signal scores each sentence, table cell and code comment so that a "
        "short instruction hidden in a long document is not diluted (it can raise WARN, never BLOCK).\n"
        "3. **Policy** - thresholds from the model card (or a labelled application override) plus rule signals give "
        "ALLOW, WARN or BLOCK; any failed check is BLOCK.\n"
        "4. **Engine 1** - only when forwarding: the text is enclosed in a fresh random boundary inside the application's "
        "trusted instructions.\n"
        "5. **Audit** - the decision is recorded (hash of the input, never the text) before anything is returned.")

    st.markdown("#### Supported integrations")
    st.markdown(
        "1. **Console** (this app): authorized people screen input for an application they are assigned to.\n"
        "2. **Python SDK** (`ppaguard.sdk.PPAGuardClient`): an approved Python application calls the same workflow with "
        "an application API key, bound to one application and revocable. A runnable example is in "
        "`examples/integration_example.py`.\n\n"
        "**Not supported**: PPA Guard is not a network proxy or HTTP API. It cannot see or stop traffic from applications "
        "that do not call it, and it does not call any LLM. Protection applies only to text submitted through these two "
        "interfaces, and only as far as the detector and the assembly defence are effective.")
    st.code(SDK, language="python")

    st.markdown("#### Role permissions (enforced on the server)")
    roles = list(Role)
    st.dataframe(pd.DataFrame([{"Permission": perm, **{ROLE_LABELS[r]: ("yes" if r in allowed else "") for r in roles}}
                               for perm, allowed in PERMISSIONS.items()]), hide_index=True, width="stretch")

"""System & Integrations: honest status of every component, the exact integration boundary, and the role matrix."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ppa_guard.engine2 import HF_REPO, HF_REVISION
from ppa_guard.platform.roles import PERMISSIONS, ROLE_LABELS, Role

from .ui import chip, header

SDK = '''# Runs inside your own Python application, on the same database as this console.
# pip install -r requirements.txt   (PPA Guard repository)
import os
from ppa_guard.platform.db import make_engine
from ppa_guard.platform.gateway import Gateway
from ppa_guard.platform.runtime import load_detector

engine = make_engine(os.environ["PPA_DATABASE_URL"])     # same Postgres as the console
detector, status = load_detector()                       # pinned ProtectAI DeBERTa v2 (ONNX, CPU)
guard = Gateway(engine, detector, status.error)

result = guard.submit_with_api_key(os.environ["PPA_GUARD_API_KEY"], untrusted_text)
if result.protected_prompt is None:          # BLOCK, or WARN held for review: do not call the LLM
    refuse(result.decision.name, result.reasons)
else:
    reply = my_llm(result.protected_prompt)  # your LLM call; PPA Guard never calls one
'''


def render(ctx) -> None:
    header("System & Integrations", "What is running, what is connected, and what PPA Guard does not do.")
    m = ctx.model
    st.markdown(
        chip(f"Engine 2: {m.model_id or 'not loaded'}", (True if m.source == "pinned-download" else None) if m.loaded else False)
        + chip(f"Provenance: {m.verified}", m.loaded and m.source == "pinned-download" or None)
        + chip("Engine 1: PPA (byte-identical to SDK b3c20a3), hardened separator choice", True)
        + chip(f"Database: {ctx.db_backend}" + (" (persistent)" if ctx.db_persistent else " (local file; resets on Streamlit Cloud restarts)"),
               True if ctx.db_persistent else None)
        + chip(f"Sign-in: {'Google / Microsoft OIDC' if ctx.mode == 'oidc' else 'DEMO personas (not authentication)'}",
               True if ctx.mode == "oidc" else None)
        + chip("LLM: not connected (placeholder adapter)", None),
        unsafe_allow_html=True)
    if m.error:
        st.error(f"Engine 2 failed to load: {m.error}. Every scan is BLOCKED until it loads (fail closed).")
    st.caption(f"Detector: {HF_REPO} at revision {HF_REVISION}, CPU via ONNX Runtime; inputs past 512 tokens are only "
               "partly seen and are flagged by policy. Pretrained only: no fine-tuning has been done in this project.")

    st.markdown("#### Supported integrations")
    st.markdown(
        "1. **Console** (this app): authorized people screen input for an application they are assigned to.\n"
        "2. **In-process Python SDK**: an approved Python application calls `Gateway.submit_with_api_key()` with an "
        "application API key, against the same database. The key is bound to one application and can be revoked.\n\n"
        "**Not supported**: PPA Guard is not a network proxy or HTTP API. It cannot see or stop traffic from applications "
        "that do not call it, and it does not call any LLM. Protection applies only to inputs that are submitted to it, "
        "and only to the extent the pretrained detector and PPA are effective on them.")
    st.code(SDK, language="python")

    st.markdown("#### Role permissions (enforced on the server)")
    roles = list(Role)
    st.dataframe(pd.DataFrame([{"Permission": perm, **{ROLE_LABELS[r]: ("yes" if r in allowed else "") for r in roles}}
                               for perm, allowed in PERMISSIONS.items()]), hide_index=True, use_container_width=True)

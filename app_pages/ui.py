"""Shared look (teal + charcoal) and small rendering helpers. Display only: no authorization decisions live here."""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timezone

import streamlit as st

TEAL, TEAL2, AMBER, RED, GREEN, MUTED = "#1FB5A3", "#2EC9B5", "#E0A23B", "#E36A5E", "#3CC48C", "#8A979D"
DECISION_COLOR = {"ALLOW": GREEN, "WARN": AMBER, "BLOCK": RED}


@dataclass
class Ctx:
    s: object            # Services
    gw: object           # Gateway
    p: object            # Principal
    org: dict
    mode: str            # oidc | demo
    model: object        # ModelStatus
    db_backend: str
    db_persistent: bool


CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.015em; }}
[data-testid="stSidebarNav"] a span {{ font-size: 0.95rem; }}
.pg-brand {{ display:flex; align-items:center; gap:10px; margin: 2px 0 14px 2px; }}
.pg-brand .lm {{ width:30px; height:30px; border-radius:8px; background:{TEAL}; display:grid; place-items:center; }}
.pg-brand b {{ font-weight:700; letter-spacing:.04em; font-size:1.02rem; }} .pg-brand i {{ font-style:normal; color:{TEAL2}; }}
.pg-sub {{ color:#9AA6AB; margin-top:-.6rem; margin-bottom:1.2rem; font-size:.95rem; }}
.kpi {{ background:#171D20; border:1px solid #252D31; border-radius:12px; padding:14px 16px; height:100%; }}
.kpi .l {{ color:#9AA6AB; font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; }}
.kpi .v {{ font-size:1.75rem; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums; }}
.kpi .x {{ color:#9AA6AB; font-size:.82rem; margin-top:2px; }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:.78rem; font-weight:600; letter-spacing:.03em;
         border:1px solid currentColor; }}
.verdict {{ border-radius:12px; padding:16px 18px; border:1px solid; display:flex; gap:16px; align-items:center; margin:.4rem 0 1rem; }}
.verdict .d {{ font-size:1.6rem; font-weight:800; letter-spacing:.04em; }}
.verdict .t {{ color:#C7D0D4; font-size:.95rem; }}
.chip {{ display:inline-flex; gap:8px; align-items:center; background:#171D20; border:1px solid #252D31; border-radius:8px;
         padding:6px 10px; margin:0 8px 8px 0; font-size:.88rem; color:#C7D0D4; }}
.dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.demo-banner {{ background:#2A2210; border:1px solid {AMBER}; color:#F3D9A4; border-radius:10px; padding:8px 12px;
                font-size:.88rem; margin-bottom:1rem; }}
code {{ font-size:.86em; }}
</style>
"""

def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def header(title: str, sub: str | None = None) -> None:
    st.markdown(f"## {title}")
    if sub:
        st.markdown(f'<div class="pg-sub">{esc(sub)}</div>', unsafe_allow_html=True)


def pill(text: str, color: str | None = None) -> str:
    color = color or DECISION_COLOR.get(text, MUTED)
    return f'<span class="pill" style="color:{color}">{esc(text)}</span>'


def kpis(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, extra) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                     f'<div class="x">{extra}</div></div>', unsafe_allow_html=True)


def verdict(decision: str, text: str) -> None:
    c = DECISION_COLOR[decision]
    st.markdown(f'<div class="verdict" style="border-color:{c};background:{c}14"><div class="d" style="color:{c}">'
                f'{esc(decision)}</div><div class="t">{esc(text)}</div></div>', unsafe_allow_html=True)


def chip(text: str, ok: bool | None) -> str:
    c = MUTED if ok is None else (GREEN if ok else RED)
    return f'<span class="chip"><span class="dot" style="background:{c}"></span>{esc(text)}</span>'


def ts(d) -> str:
    if d is None:
        return ""
    if isinstance(d, datetime):
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(d)


def ms(x) -> str:
    return "n/a" if x is None else (f"{x:.0f} ms" if x < 1000 else f"{x / 1000:.2f} s")

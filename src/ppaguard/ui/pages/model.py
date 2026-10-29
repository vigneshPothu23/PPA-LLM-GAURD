"""Engine 2 Model: the active model card - provenance, calibration, thresholds and the measured evaluation results.
Everything shown here is read from the model card written by the training/evaluation pipeline; nothing is typed in."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.roles import has
from ..theme import AMBER, RED, chip, header, kpis, pill


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render(ctx) -> None:
    if not has(ctx.p.role, "model.view"):
        st.error("Not permitted.")
        return
    header("Engine 2 Model", "The detector in use, where it came from, and how it was evaluated.")
    m = ctx.model
    if not m.loaded:
        st.error(f"No model loaded: {m.error}. Every scan is BLOCKED (fail closed).")
        return
    det = ctx.wf.detector
    card = det.card
    extra = card.extra
    prov_color = {"test-only": RED, "trained-from-scratch-ppaguard": AMBER}.get(card.provenance)
    st.markdown(chip(card.model_id, None if card.provenance == "test-only" else True)
                + (pill(m.detail.get("provenance_text", card.provenance), prov_color) if prov_color
                   else chip(m.detail.get("provenance_text", card.provenance), True)), unsafe_allow_html=True)
    if card.provenance == "test-only":
        st.warning("This is a test artifact with random weights. Its scores carry no meaning.")
    elif extra.get("status") != "validated":
        st.warning("**EXPERIMENTAL.** " + (extra.get("limitations") or
                   "No evaluation on the frozen test set supports production use."))
    if card.provenance == "trained-from-scratch-ppaguard":
        st.info("Trained from random initialisation on the audited training set (no pretrained weights were available in "
                "the build environment). A pretrained fine-tuned model can replace it without code changes.")

    t = card.thresholds
    kpis([("WARN if P >", f"{t['warn']:.2f}", t.get("source", "")), ("BLOCK if P >", f"{t['block']:.2f}", t.get("source", "")),
          ("Temperature", f"{card.temperature:.3f}", "calibration (1.0 = none)"),
          ("Max length", str(card.max_length), f"windows of {card.max_length}, up to {card.max_windows}")])
    ctxsig = getattr(det, "context_signal", None)
    st.markdown("#### Context signal (hidden instructions inside documents)")
    if ctxsig is None:
        st.warning("Context signal disabled (PPAGUARD_CONTEXT=off): only the whole-document score is used.")
    else:
        kpis([("Context WARN if P >", f"{t.get('context_warn', 0.95):.2f}", "WARN only - never BLOCK"),
              ("Segments", "sentences · table cells · code comments", "each scored separately"),
              ("Features", "10", "lexical cues, Engine 2 on the segment, off-topic distance (MiniLM-L6)"),
              ("Model", "logistic regression", "artifacts/context_guard/guard_full.json")])
        st.caption(f"{ctxsig.model_id}. A whole-document score dilutes a short instruction hidden in a long README, e-mail, "
                   "table or log; the context signal scores each segment and holds the input for review when one looks like "
                   "an instruction aimed at the AI. Selected on development data only; see docs/v5/FINAL_EVALUATION.md "
                   "for its measured holdout results and remaining errors.")

    metrics = extra.get("metrics") or {}
    rows = []
    for split in ("dev", "test"):
        s = metrics.get(split)
        if not s:
            continue
        for lvl in ("warn", "block"):
            a = s[lvl]
            rows.append({"Set": f"frozen {split}" + (" (thresholds chosen here)" if split == "dev" else " (evaluated once)"),
                         "Rows": s["n"], "ROC-AUC": round(s["roc_auc"], 4) if s["roc_auc"] is not None else None,
                         "Avg precision": round(s["average_precision"], 4) if s["average_precision"] is not None else None,
                         "Level": f"{lvl.upper()} > {a['threshold']:.2f}", "Injection recall": _pct(a["injection_recall"]),
                         "Injection precision": _pct(a["injection_precision"]), "False-positive rate": _pct(a["fpr"])})
    st.markdown("#### Evaluation (measured by the pipeline, not typed in)")
    st.caption("Whole-document detector alone, at the thresholds in each row (the earlier WARN 0.01 configuration). "
               "Results of the final configuration (detector + context signal): docs/v5/FINAL_EVALUATION.md.")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("This model card carries no evaluation results. Run `python -m ppaguard.engine2.evaluate` to produce them.")

    if extra.get("threshold_notes"):
        st.markdown("**Threshold notes from the selection rule**")
        for n in extra["threshold_notes"]:
            st.caption(n)
    for rev in extra.get("revisions", []):
        st.caption(f"Revision {rev.get('utc', '')}: {rev.get('reason', '')}")
    if extra.get("base_provenance"):
        st.caption("Pretrained base: " + extra["base_provenance"].get("source", "") + " · licence "
                   + str(extra["base_provenance"].get("licence", "")) + " · " + extra["base_provenance"].get("note", ""))

    with st.expander("Training, data and artifact details"):
        tr = extra.get("training") or {}
        data = extra.get("data") or {}
        info = {"architecture": extra.get("architecture"), "parameters": extra.get("parameters"), "base model": extra.get("base_model"),
                "created (UTC)": extra.get("created_utc"), "training stopped": tr.get("stopped"),
                "best internal-val step": (tr.get("best") or {}).get("step"),
                "train rows": (data.get("train") or {}).get("rows"), "train injection share": (data.get("train") or {}).get("injection_share"),
                "train file sha256": data.get("train_file_sha256"), "leakage overlaps with frozen sets": data.get("leakage_overlaps"),
                "quantization": (extra.get("quantization") or {}).get("shipped"), "files verified at load": bool(card.files)}
        st.json({k: v for k, v in info.items() if v is not None}, expanded=True)
        if extra.get("source"):
            st.json(extra["source"], expanded=False)

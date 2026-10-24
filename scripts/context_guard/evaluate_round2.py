"""Round 2: fit the document-aware second stage on development data, then (once) score the round-2 test sets.

    python scripts/context_guard/evaluate_round2.py --dev-only   # fit, CV, thresholds, write artifact; no test data
    python scripts/context_guard/evaluate_round2.py              # score test sets; REFUSES unless the artifact's
                                                                 # SHA-256 is recorded in PREREGISTRATION_R2.md

Protocol
  * stage 1 = the round-1 segment guard, two views (all features / no off-topic), fitted on ctx_dev exactly as in
    round 1;
  * stage-2 development set = ctx_dev (with OUT-OF-FOLD stage-1 scores, so stage 2 never sees optimistic in-sample
    stage-1 scores) + dev2_email;
  * stage-2 thresholds are selected on OUT-OF-FOLD stage-2 scores (5-fold, stratified, seed 0) with the same
    FPR-bounded rule as everything else; the final stage-2 model is then fitted on the whole development set;
  * every comparison model gets its thresholds from the SAME stage-2 development set, so all are tuned on identical
    data; ProtectAI is also shown at the app's model-card thresholds (0.01 / 0.99) for reference;
  * test sets for this round: unseen_email_3, unseen_embed_3 (clean). Earlier sets are reported too but are labelled
    development-influenced, because round 2 was designed after looking at them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import evaluate_from_cache as E  # noqa: E402
from ppaguard.engine2 import metrics as M  # noqa: E402
from ppaguard.experimental.context_guard import stage2 as S2  # noqa: E402

ART = ROOT / "artifacts/context_guard/two_stage.json"
PREREG = ROOT / "docs/v5/evidence/context_guard/PREREGISTRATION_R2.md"
OUT = ROOT / "docs/v5/evidence/context_guard"
TEST_SETS = ["unseen_email_3", "unseen_embed_3"]
EARLIER = ["frozen_test", "unseen_synth", "unseen_email", "unseen_email_2", "frozen_dev"]
BASELINES = ["protectai", "roberta6l", "scratch", "deepset", "fmops"]
NOOFF = (E.OFF_COL,)


def folds_for(y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.zeros(len(y), dtype=int)
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        f[idx] = np.arange(len(idx)) % k
    return f


def stage1_scores(rows, m_full, m_nooff):
    return E.doc_scores(m_full, rows, ()), E.doc_scores(m_nooff, rows, NOOFF)


def doc_matrix(rows, g, gn):
    return np.asarray([S2.doc_features(g[i], gn[i], r["wholedoc"]["protectai"], r["kinds"]) for i, r in enumerate(rows)])


def build_dev():
    ctx, dev2 = E.load("ctx_dev"), E.load("dev2_email")
    m_full, m_nooff = E.fit_ablation(ctx, ()), E.fit_ablation(ctx, NOOFF)
    # out-of-fold stage-1 scores for ctx_dev rows
    yc = np.array([r["label"] for r in ctx])
    f = folds_for(yc)
    g_c, gn_c = np.zeros(len(ctx)), np.zeros(len(ctx))
    for k in range(5):
        tr = [r for r, ff in zip(ctx, f) if ff != k]
        te = np.where(f == k)[0]
        mf, mn = E.fit_ablation(tr, ()), E.fit_ablation(tr, NOOFF)
        sub = [ctx[i] for i in te]
        g_c[te], gn_c[te] = E.doc_scores(mf, sub, ()), E.doc_scores(mn, sub, NOOFF)
    g_d, gn_d = stage1_scores(dev2, m_full, m_nooff)
    rows = ctx + dev2
    X = np.vstack([doc_matrix(ctx, g_c, gn_c), doc_matrix(dev2, g_d, gn_d)])
    y = np.array([r["label"] for r in rows])
    stage1_dev = np.concatenate([g_c, g_d])
    return rows, X, y, m_full, m_nooff, stage1_dev


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def thresholds(y, p):
    s = M.select_thresholds(y, p)
    return {"WARN": s["warn"], "BLOCK": s["block"]}, s.get("notes")


def dev_only():
    rows, X, y, m_full, m_nooff, g_dev = build_dev()
    f = folds_for(y)
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        oof[te] = S2.fit_stage2(X[tr], y[tr]).predict_proba(X[te])
    final = S2.fit_stage2(X, y)
    th, notes = thresholds(y, oof)
    pa = np.array([r["wholedoc"]["protectai"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    base_th = {}
    for b in BASELINES:
        base_th[b] = thresholds(y, np.array([r["wholedoc"][b] for r in rows]))[0]
    base_th["guard_full"] = thresholds(y, g_dev)[0]
    cv = {"two_stage_oof_roc_auc": round(M.roc_auc(y, oof), 4), "protectai_roc_auc": round(M.roc_auc(y, pa), 4),
          "guard_full_stage1_roc_auc": round(M.roc_auc(y, g_dev), 4)}
    for gname in sorted(set(groups)):
        m = groups == gname
        cv[f"group:{gname}"] = {"n": int(m.sum()), "pos": int(y[m].sum()),
                                "two_stage_oof_flag_rate_at_WARN": round(float((oof[m] > th["WARN"]).mean()), 4),
                                "protectai_flag_rate_at_its_WARN": round(float((pa[m] > base_th["protectai"]["WARN"]).mean()), 4)}
    md = lambda m: {"feature_names": m.feature_names, "mean": m.mean, "std": m.std, "weights": m.weights, "bias": m.bias}  # noqa: E731
    art = {"name": "context-guard-two-stage", "status": "EXPERIMENTAL - not validated for production",
           "stage1_full": md(m_full), "stage1_nooff": md(m_nooff), "stage2": md(final), "offtopic_col": E.OFF_COL,
           "thresholds_dev_oof": th, "threshold_notes": notes, "baseline_thresholds_same_dev": base_th,
           "development_set": {"ctx_dev_sha256": sha(ROOT / "eval/context_sets/ctx_dev.jsonl"),
                               "dev2_email_sha256": sha(ROOT / "eval/context_sets/dev2_email.jsonl"),
                               "rows": int(len(y)), "injected": int(y.sum())},
           "dev_cross_validation": cv,
           "stage2_weights": dict(zip(S2.FEATURES, [round(w, 4) for w in final.weights]))}
    ART.write_text(json.dumps(art, indent=2))
    print("wrote", ART, "sha256", sha(ART))
    print(json.dumps({"cv": cv, "thresholds": th, "notes": notes, "baseline_thresholds": base_th,
                      "weights": art["stage2_weights"]}, indent=1))


def score_tests():
    if not PREREG.exists() or sha(ART) not in PREREG.read_text():
        raise SystemExit("refusing to score: artifact SHA-256 is not recorded in PREREGISTRATION_R2.md")
    art = json.loads(ART.read_text())
    m_full, m_nooff, st2 = (S2.TwoStageGuard._m(art[k]) for k in ("stage1_full", "stage1_nooff", "stage2"))
    th = dict(art["baseline_thresholds_same_dev"])
    th["two_stage"] = art["thresholds_dev_oof"]
    th["protectai_card"] = {"WARN": 0.01, "BLOCK": 0.99}
    res = {"artifact_sha256": sha(ART), "thresholds": th, "sets": {}}
    for s in TEST_SETS + EARLIER:
        if not E.complete(s):
            raise SystemExit(f"cache for {s} incomplete")
        rows = E.load(s)
        y = np.array([r["label"] for r in rows])
        g, gn = stage1_scores(rows, m_full, m_nooff)
        scores = {"two_stage": st2.predict_proba(doc_matrix(rows, g, gn)), "guard_full": g}
        for b in BASELINES:
            scores[b] = np.array([r["wholedoc"][b] for r in rows])
        scores["protectai_card"] = scores["protectai"]
        carriers = [r["carrier"] for r in rows]
        res["sets"][s] = {m: E.evaluate(y, p, carriers, {**th[m], "0.5": 0.5}) for m, p in scores.items()}
        res["sets"][s]["_clean_test"] = s in TEST_SETS
    # pooled round-2 test (both sets) at WARN, for H6
    pooled = {}
    rows = E.load(TEST_SETS[0]) + E.load(TEST_SETS[1])
    y = np.array([r["label"] for r in rows])
    g, gn = stage1_scores(rows, m_full, m_nooff)
    p2 = st2.predict_proba(doc_matrix(rows, g, gn))
    pa = np.array([r["wholedoc"]["protectai"] for r in rows])
    pooled["two_stage"] = E.evaluate(y, p2, ["pooled"] * len(y), {**th["two_stage"], "0.5": 0.5})
    pooled["protectai"] = E.evaluate(y, pa, ["pooled"] * len(y), {**th["protectai"], "0.5": 0.5})
    res["pooled_round2_test"] = pooled
    a = lambda s, m: res["sets"][s][m]["roc_auc"]  # noqa: E731
    w2, wp = pooled["two_stage"]["at"]["WARN"], pooled["protectai"]["at"]["WARN"]
    res["hypotheses"] = {
        "H4_email_auc_noninferior": {"two_stage": a("unseen_email_3", "two_stage"), "protectai": a("unseen_email_3", "protectai"),
                                     "holds": a("unseen_email_3", "two_stage") >= a("unseen_email_3", "protectai") - 0.02},
        "H5_embedded_auc_superior": {"two_stage": a("unseen_embed_3", "two_stage"), "protectai": a("unseen_embed_3", "protectai"),
                                     "holds": a("unseen_embed_3", "two_stage") > a("unseen_embed_3", "protectai")},
        "H6_pooled_warn": {"two_stage": {"recall": w2["recall"], "fpr": w2["fpr"]},
                           "protectai": {"recall": wp["recall"], "fpr": wp["fpr"]},
                           "holds": (w2["fpr"] <= wp["fpr"]) and (w2["recall"] >= wp["recall"] - 0.05)},
    }
    res["hypotheses"]["improvement_validated"] = all(v["holds"] for v in res["hypotheses"].values())
    (OUT / "comparison_round2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["hypotheses"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-only", action="store_true")
    a = ap.parse_args()
    dev_only() if a.dev_only else score_tests()

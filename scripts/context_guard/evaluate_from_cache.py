"""Fit, threshold and evaluate the context guard and baselines from the cached feature matrices.

Reads docs/v5/evidence/context_guard/cache/<set>.jsonl (produced by cache_features.py) and does only cheap
numpy work:
  * guard ablations are column subsets of the one cached feature matrix:
      guard_full        all columns
      guard_no_offtopic offtopic column zeroed
      guard_no_detector detector_score column zeroed
      guard_cues_only   both zeroed
  * each ablation's LR is fitted on ctx_dev with SEGMENT-level labels (seg_pos);
  * component justification on DEVELOPMENT data only: 5-fold cross-validation on ctx_dev
    (out-of-fold ROC-AUC / AP per ablation);
  * thresholds (WARN/BLOCK) are selected on ctx_dev with the FPR-bounded rule and applied to every set;
  * baselines use the cached whole-document scores;
  * hybrid_max = max(guard_full, protectai_wholedoc): POST HOC (defined after the first unseen e-mail
    result; see PREREGISTRATION.md) and labelled as such;
  * 95% bootstrap confidence intervals (1000 resamples, seed 0) for ROC-AUC and WARN recall / FPR;
  * subsets: carrier-level (frozen sets: the `set` field; others: the `group` field).

Writes comparison.json. Render with report_context.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ppaguard.engine2 import metrics as M  # noqa: E402
from ppaguard.experimental.context_guard.cues import FEATURE_NAMES  # noqa: E402
from ppaguard.experimental.context_guard.fusion import fit_logistic  # noqa: E402

CACHE = ROOT / "docs/v5/evidence/context_guard/cache"
OUT = ROOT / "docs/v5/evidence/context_guard"
FEATURES = FEATURE_NAMES + ["detector_score", "offtopic", "kind_weight"]
DET_COL = FEATURES.index("detector_score")
OFF_COL = FEATURES.index("offtopic")
SET_ORDER = ["frozen_dev", "frozen_test", "unseen_synth", "unseen_email", "unseen_email_2"]
SOURCE = {"frozen_dev": ROOT / "eval/sets/validation.jsonl", "frozen_test": ROOT / "eval/sets/test.jsonl"}
ABLATIONS = {
    "guard_full": (),
    "guard_no_offtopic": (OFF_COL,),
    "guard_no_detector": (DET_COL,),
    "guard_cues_only": (OFF_COL, DET_COL),
}
POST_HOC = {"hybrid_max"}


def load(set_name: str) -> list[dict]:
    with open(CACHE / f"{set_name}.jsonl") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    if set_name in SOURCE:  # frozen rows have no id: cache order == file order; take carrier from `set`
        with open(SOURCE[set_name]) as fh:
            src = [json.loads(l) for l in fh if l.strip()]
        assert len(src) == len(rows)
        for r, s in zip(rows, src):
            assert r["label"] == int(s["label"])
            r["carrier"] = s["set"]
    else:
        for r in rows:
            r["carrier"] = r["group"]
    return rows


def mask_X(X: np.ndarray, zero_cols: tuple[int, ...]) -> np.ndarray:
    if not zero_cols:
        return X
    X = X.copy()
    X[:, list(zero_cols)] = 0.0
    return X


def fit_ablation(rows: list[dict], zero_cols: tuple[int, ...]):
    Xs, ys = [], []
    for r in rows:
        X = np.asarray(r["X"], dtype=np.float64)
        if X.size == 0:
            continue
        X = mask_X(X, zero_cols)
        pos = set(r["seg_pos"])
        for i in range(len(X)):
            Xs.append(X[i])
            ys.append(1 if (r["label"] == 1 and i in pos) else 0)
    return fit_logistic(np.asarray(Xs), np.asarray(ys), FEATURES, l2=1.0, seed=0)


def doc_scores(model, rowset: list[dict], zero_cols: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(len(rowset))
    for k, r in enumerate(rowset):
        X = np.asarray(r["X"], dtype=np.float64)
        out[k] = 0.0 if X.size == 0 else float(model.predict_proba(mask_X(X, zero_cols)).max())
    return out


def _r(x):
    return round(float(x), 4) if x is not None else None


def mblock(y, p, t):
    r = M.report_at(y, p, t)
    i, c = r["injection"], r["confusion"]
    return {"threshold": round(float(t), 4), "precision": _r(i["precision"]), "recall": _r(i["recall"]),
            "f1": _r(i["f1"]), "fpr": _r(r["false_positive_rate"]), "tp": c["tp"], "fp": c["fp"],
            "tn": c["tn"], "fn": c["fn"], "n": int(len(y)), "pos": int(y.sum())}


def bootstrap_ci(y, p, t, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, recs, fprs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if 0 < yb.sum() < n:
            aucs.append(M.roc_auc(yb, pb))
        pred = pb > t
        if yb.sum():
            recs.append((pred & (yb == 1)).sum() / yb.sum())
        if (yb == 0).sum():
            fprs.append((pred & (yb == 0)).sum() / (yb == 0).sum())
    q = lambda a: [_r(np.percentile(a, 2.5)), _r(np.percentile(a, 97.5))] if a else None  # noqa: E731
    return {"roc_auc": q(aucs), "warn_recall": q(recs), "warn_fpr": q(fprs)}


def evaluate(y, p, carriers, th):
    both = 0 < y.sum() < len(y)
    out = {"n": int(len(y)), "pos": int(y.sum()),
           "roc_auc": _r(M.roc_auc(y, p)) if both else None,
           "average_precision": _r(M.average_precision(y, p)) if both else None,
           "at": {k: mblock(y, p, t) for k, t in th.items()},
           "ci95": bootstrap_ci(y, p, th["WARN"]), "subsets": {}}
    g = np.array(carriers)
    for s in sorted(set(carriers)):
        m = g == s
        ys, ps = y[m], p[m]
        sub_both = 0 < ys.sum() < m.sum()
        out["subsets"][s] = {"n": int(m.sum()), "pos": int(ys.sum()),
                             "roc_auc": _r(M.roc_auc(ys, ps)) if sub_both else None,
                             "at": {k: mblock(ys, ps, t) for k, t in th.items()}}
    return out


def cross_validate(ctx: list[dict], k: int = 5, seed: int = 0) -> dict:
    """Out-of-fold document scores on ctx_dev for each ablation (stratified by label)."""
    rng = np.random.default_rng(seed)
    y = np.array([r["label"] for r in ctx])
    folds = np.zeros(len(ctx), dtype=int)
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    res = {}
    for name, zc in ABLATIONS.items():
        oof = np.zeros(len(ctx))
        for f in range(k):
            train = [r for r, ff in zip(ctx, folds) if ff != f]
            test_idx = np.where(folds == f)[0]
            m = fit_ablation(train, zc)
            oof[test_idx] = doc_scores(m, [ctx[i] for i in test_idx], zc)
        res[name] = {"oof_roc_auc": _r(M.roc_auc(y, oof)), "oof_average_precision": _r(M.average_precision(y, oof))}
    res["protectai_wholedoc"] = {"oof_roc_auc": _r(M.roc_auc(y, np.array([r["wholedoc"]["protectai"] for r in ctx]))),
                                 "oof_average_precision": _r(M.average_precision(
                                     y, np.array([r["wholedoc"]["protectai"] for r in ctx]))),
                                 "note": "no fitting; whole-document scores"}
    return res


INPUT = {"ctx_dev": ROOT / "eval/context_sets/ctx_dev.jsonl",
         "frozen_dev": ROOT / "eval/sets/validation.jsonl", "frozen_test": ROOT / "eval/sets/test.jsonl",
         "unseen_synth": ROOT / "eval/context_sets/unseen_synth.jsonl",
         "unseen_email": ROOT / "eval/context_sets/unseen_email.jsonl",
         "unseen_email_2": ROOT / "eval/context_sets/unseen_email_2.jsonl",
         "dev2_email": ROOT / "eval/context_sets/dev2_email.jsonl",
         "unseen_email_3": ROOT / "eval/context_sets/unseen_email_3.jsonl",
         "unseen_embed_3": ROOT / "eval/context_sets/unseen_embed_3.jsonl"}


def complete(set_name: str) -> bool:
    """A set is scored only when its cache covers every input row (never a partial cache)."""
    c, src = CACHE / f"{set_name}.jsonl", INPUT[set_name]
    if not c.exists() or not src.exists():
        return False
    n = lambda p: sum(1 for l in open(p) if l.strip())  # noqa: E731
    ok = n(c) == n(src)
    if not ok:
        print(f"SKIP {set_name}: cache incomplete ({n(c)}/{n(src)} rows)")
    return ok


def main():
    sets = {s: load(s) for s in ["ctx_dev"] + SET_ORDER if complete(s)}
    ctx = sets["ctx_dev"]
    baseline_names = sorted(ctx[0]["wholedoc"].keys())

    models = {}
    for name, zc in ABLATIONS.items():
        models[name] = ("guard", fit_ablation(ctx, zc), zc)
    for bn in baseline_names:
        models["protectai_wholedoc" if bn == "protectai" else bn] = ("baseline", bn)
    models["hybrid_max"] = ("hybrid",)

    def score(name, rowset):
        spec = models[name]
        if spec[0] == "guard":
            return doc_scores(spec[1], rowset, spec[2])
        if spec[0] == "baseline":
            return np.array([r["wholedoc"][spec[1]] for r in rowset])
        return np.maximum(score("guard_full", rowset), score("protectai_wholedoc", rowset))

    yctx = np.array([r["label"] for r in ctx])
    thresholds, th_notes = {}, {}
    for name in models:
        sel = M.select_thresholds(yctx, score(name, ctx))
        thresholds[name] = {"WARN": sel["warn"], "BLOCK": sel["block"], "0.5": 0.5}
        th_notes[name] = sel.get("notes")

    results = {"protocol": "fit + thresholds on ctx_dev only; every other set scored with fixed models/thresholds",
               "post_hoc_models": sorted(POST_HOC),
               "ctx_dev_n": len(ctx), "ctx_dev_pos": int(yctx.sum()),
               "cross_validation_ctx_dev": cross_validate(ctx),
               "thresholds": thresholds, "threshold_notes": th_notes,
               "guard_weights": {n: dict(zip(FEATURES, [round(w, 4) for w in models[n][1].weights]))
                                 for n in ABLATIONS},
               "sets": {}}
    for sname in SET_ORDER:
        if sname not in sets:
            continue
        rowset = sets[sname]
        y = np.array([r["label"] for r in rowset])
        carriers = [r["carrier"] for r in rowset]
        results["sets"][sname] = {m: evaluate(y, score(m, rowset), carriers, thresholds[m]) for m in models}

    (OUT / "comparison.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUT / "comparison.json")
    print("CV on ctx_dev:", json.dumps(results["cross_validation_ctx_dev"]))
    for sname in results["sets"]:
        print(f"{sname}: " + " ".join(f"{m}={results['sets'][sname][m]['roc_auc']}" for m in results["sets"][sname]))


if __name__ == "__main__":
    main()

"""Phase C design selection on DEVELOPMENT data only (nested, grouped 5-fold cross-validation).

Development pool (no holdout set is read):
  frozen_dev (deepset direct prompts + NotInject_one), ctx_dev, dev2_email, bipia_train_dev (BIPIA TRAIN contexts),
  scenarios_dev (synthetic suite, dev split).

Candidate Engine 2 designs (decision = WARN/BLOCK vs ALLOW; scored as "flagged"):
  A  shipped:     whole-document ProtectAI, thresholds of the shipped app (WARN > 0.01, BLOCK > 0.99)
  B  recal:       whole-document ProtectAI, thresholds re-selected on the training folds
  C  segmax:      max(whole-document, per-segment ProtectAI), thresholds re-selected
  D  or_guard_r1: whole-document ProtectAI OR the shipped round-1 context guard (fitted on ctx_dev), joint thresholds
  E  or_guard_cv: whole-document ProtectAI OR a context guard re-fitted on the training folds, joint thresholds

Threshold rule inside each training fold (the project's FPR-bounded rule):
  BLOCK: whole-document score only; lowest t with precision >= 0.98 and FPR <= 1 % at t and above (else 0.99 + note)
  WARN : maximise recall subject to pooled FPR <= 10 % (joint grid for the OR designs; tie -> lower FPR)
The held-out fold is scored with those thresholds, giving honest out-of-fold (OOF) decisions for every variant.

Pre-stated selection criterion: highest OOF Youden index (recall - FPR) on the pooled development data; a more complex
variant is preferred only if it beats the simpler one by more than the larger of the two bootstrap half-widths.
"""
from __future__ import annotations

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
from ppaguard.experimental.context_guard.fusion import LogisticModel  # noqa: E402

DEV_SETS = ["frozen_dev", "ctx_dev", "dev2_email", "bipia_train_dev", "scenarios_dev"]
GRID = np.unique(np.concatenate([np.round(np.linspace(0.01, 0.99, 99), 2), [0.995, 0.999]]))
OUT = ROOT / "docs/v5/evidence/final"


def groups_for(set_name: str, rows: list[dict]) -> list[str]:
    src = E.INPUT[set_name]
    if set_name == "frozen_dev":
        return [f"fd{i}" for i in range(len(rows))]
    by_id = {}
    with open(src, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                r = json.loads(line)
                by_id[r.get("id") or f"row{i}"] = r
    out = []
    for r in rows:
        s = by_id[r["id"]]
        out.append(f"{set_name}:" + str(s.get("context_id") or s.get("carrier") or r["id"]))
    return out


def fold_of(groups: list[str], labels: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    assign = {g: i % k for i, g in enumerate(uniq)}
    return np.array([assign[g] for g in groups])


def segmax(rows):
    return np.array([max(r["wholedoc"]["protectai"], float(np.max(np.asarray(r["X"])[:, E.DET_COL])) if r["X"] else 0.0)
                     for r in rows])


def warn_single(y, s):
    best = (-1, 1, 0.99)
    for t in GRID:
        pred = s > t
        fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        rec = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        if fpr <= 0.10 and (rec > best[0] or (rec == best[0] and fpr < best[1])):
            best = (rec, fpr, float(t))
    return best[2]


def warn_joint(y, s1, s2):
    best = (-1, 1, 0.99, 0.99)
    for t1 in GRID:
        p1 = s1 > t1
        for t2 in GRID:
            pred = p1 | (s2 > t2)
            fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
            if fpr > 0.10:
                continue
            rec = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
            if rec > best[0] or (rec == best[0] and fpr < best[1]):
                best = (rec, fpr, float(t1), float(t2))
    return best[2], best[3]


def block_t(y, s):
    sel = M.select_thresholds(y, s)
    return sel["block"]


def summary(y, flagged, blocked):
    tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum()); tn = int((~flagged & (y == 0)).sum())
    rec = tp / max(1, tp + fn); fpr = fp / max(1, fp + tn); prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": round(rec, 4), "fpr": round(fpr, 4),
            "precision": round(prec, 4) if prec is not None else None, "f1": round(f1, 4), "youden": round(rec - fpr, 4),
            "blocked_benign": int((blocked & (y == 0)).sum()), "blocked_attack": int((blocked & (y == 1)).sum())}


def boot_youden(y, flagged, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yb, fb = y[i], flagged[i]
        if (yb == 1).any() and (yb == 0).any():
            vals.append((fb & (yb == 1)).sum() / (yb == 1).sum() - (fb & (yb == 0)).sum() / (yb == 0).sum())
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main():
    rows, names, groups = [], [], []
    for s in DEV_SETS:
        if not E.complete(s):
            raise SystemExit(f"cache for {s} incomplete")
        r = E.load(s)
        rows += r
        names += [s] * len(r)
        groups += groups_for(s, r)
    y = np.array([r["label"] for r in rows])
    names = np.array(names)
    folds = fold_of(groups, y)
    pdoc = np.array([r["wholedoc"]["protectai"] for r in rows])
    pseg = segmax(rows)
    art = json.loads((ROOT / "artifacts/context_guard/guard_full.json").read_text())["model"]
    g_r1_model = LogisticModel(art["feature_names"], art["mean"], art["std"], art["weights"], art["bias"])
    g_r1 = E.doc_scores(g_r1_model, rows, ())

    variants = ["A_shipped", "B_recal", "C_segmax", "D_or_guard_r1", "E_or_guard_cv"]
    flagged = {v: np.zeros(len(y), bool) for v in variants}
    blocked = {v: np.zeros(len(y), bool) for v in variants}
    fold_thr = {v: [] for v in variants}
    g_cv = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        # A: fixed shipped thresholds
        flagged["A_shipped"][te] = pdoc[te] > 0.01
        blocked["A_shipped"][te] = pdoc[te] > 0.99
        tb = block_t(y[tr], pdoc[tr])
        # B
        tw = warn_single(y[tr], pdoc[tr])
        flagged["B_recal"][te] = (pdoc[te] > tw) | (pdoc[te] > tb)
        blocked["B_recal"][te] = pdoc[te] > tb
        fold_thr["B_recal"].append({"warn": tw, "block": tb})
        # C
        tw = warn_single(y[tr], pseg[tr])
        flagged["C_segmax"][te] = (pseg[te] > tw) | (pdoc[te] > tb)
        blocked["C_segmax"][te] = pdoc[te] > tb
        fold_thr["C_segmax"].append({"warn": tw, "block": tb})
        # D
        t1, t2 = warn_joint(y[tr], pdoc[tr], g_r1[tr])
        flagged["D_or_guard_r1"][te] = (pdoc[te] > t1) | (g_r1[te] > t2) | (pdoc[te] > tb)
        blocked["D_or_guard_r1"][te] = pdoc[te] > tb
        fold_thr["D_or_guard_r1"].append({"warn_doc": t1, "warn_guard": t2, "block": tb})
        # E: refit guard on training folds (segment labels only where the injected payload is known)
        train_rows = [r for r, m in zip(rows, tr) if m and (r["label"] == 0 or r["seg_pos"])]
        gm = E.fit_ablation(train_rows, ())
        s_tr, s_te = E.doc_scores(gm, [r for r, m in zip(rows, tr) if m], ()), E.doc_scores(gm, [r for r, m in zip(rows, te) if m], ())
        g_cv[te] = s_te
        t1, t2 = warn_joint(y[tr], pdoc[tr], s_tr)
        flagged["E_or_guard_cv"][te] = (pdoc[te] > t1) | (s_te > t2) | (pdoc[te] > tb)
        blocked["E_or_guard_cv"][te] = pdoc[te] > tb
        fold_thr["E_or_guard_cv"].append({"warn_doc": t1, "warn_guard": t2, "block": tb})
        print(f"fold {k} done", flush=True)

    res = {"dev_sets": DEV_SETS, "n": int(len(y)), "positives": int(y.sum()), "variants": {}}
    for v in variants:
        res["variants"][v] = {"oof_pooled": summary(y, flagged[v], blocked[v]),
                              "youden_ci95": boot_youden(y, flagged[v]),
                              "per_set": {s: summary(y[names == s], flagged[v][names == s], blocked[v][names == s]) for s in DEV_SETS},
                              "fold_thresholds": fold_thr[v]}
    res["threshold_free_auc"] = {"pdoc": round(M.roc_auc(y, pdoc), 4), "segmax": round(M.roc_auc(y, pseg), 4),
                                 "guard_r1": round(M.roc_auc(y, g_r1), 4), "guard_cv_oof": round(M.roc_auc(y, g_cv), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_selection_cv.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["threshold_free_auc"]))
    for v in variants:
        o = res["variants"][v]["oof_pooled"]
        print(f"{v:15s} recall {o['recall']:.3f} fpr {o['fpr']:.3f} prec {o['precision']} f1 {o['f1']:.3f} youden {o['youden']:.3f} "
              f"{res['variants'][v]['youden_ci95']} blocked_benign {o['blocked_benign']}")
        for s in DEV_SETS:
            p = res["variants"][v]["per_set"][s]
            print(f"     {s:16s} R {p['recall']:.3f} FPR {p['fpr']:.3f} (tp {p['tp']} fp {p['fp']} fn {p['fn']} tn {p['tn']})")


if __name__ == "__main__":
    main()

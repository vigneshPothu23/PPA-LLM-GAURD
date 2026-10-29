"""Compare two run_pipeline_eval.py outputs (baseline vs final) on identical inputs. Written before the holdout
results were read; the analysis below is fixed in advance.

    python eval/compare_pipeline_runs.py --baseline DIR --final DIR --out docs/v5/evidence/final/holdout

Per set: confusion matrix, precision, recall, F1, FPR, Youden index (recall - FPR, the pre-registered criterion),
paired bootstrap 95 % CI of the Youden difference (rows resampled jointly, 2,000 draws, seed 0), exact McNemar test on
rows where exactly one configuration is correct, blocked-benign counts, latency, how many final WARNs came only from
the context signal, per-category results, and the ids of every remaining error. Writes comparison.json and
comparison.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(d: Path, name: str) -> dict:
    return {r["id"]: r for r in map(json.loads, (d / f"predictions_{name}.jsonl").read_text().splitlines()) if r}


def stats(y: np.ndarray, f: np.ndarray) -> dict:
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    f1 = (2 * prec * rec / (prec + rec) if prec and rec else 0.0) if prec is not None and rec is not None else None
    you = rec - fpr if rec is not None and fpr is not None else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "youden": you}


def youden(y, f):
    pos, neg = y == 1, y == 0
    return f[pos].mean() - f[neg].mean()


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(base: dict, fin: dict) -> dict:
    ids = sorted(base)
    assert set(ids) == set(fin), "baseline and final scored different rows"
    y = np.array([base[i]["label"] for i in ids])
    fb = np.array([base[i]["flagged"] for i in ids])
    ff = np.array([fin[i]["flagged"] for i in ids])
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        s = rng.integers(0, len(ids), len(ids))
        if (y[s] == 1).any() and (y[s] == 0).any():
            diffs.append(youden(y[s], ff[s]) - youden(y[s], fb[s]))
    cb, cf = fb == (y == 1), ff == (y == 1)
    b_only, f_only = int((cb & ~cf).sum()), int((~cb & cf).sum())

    def lat(d):
        v = sorted(d[i]["latency_ms"] for i in ids)
        return {"mean": round(float(np.mean(v)), 1), "p95": round(v[max(0, math.ceil(0.95 * len(v)) - 1)], 1)}

    def blocked_benign(d):
        return sum(1 for i in ids if d[i]["label"] == 0 and d[i]["decision"] == "BLOCK")

    ctx_only = [i for i in ids if fin[i]["decision"] == "WARN"
                and all(r.startswith("[engine2-context]") or r.split("->")[-1].strip() == "ALLOW"
                        for r in fin[i]["reasons"])]
    cats = defaultdict(list)
    for k, i in enumerate(ids):
        cats[(base[i]["split"], base[i]["category"], int(y[k]))].append(k)
    per_cat = {}
    for (sp, c, lab), ks in sorted(cats.items()):
        ks = np.array(ks)
        per_cat[f"{sp}|{c}|{'attack' if lab else 'benign'}"] = {
            "n": len(ks), "baseline_flagged": int(fb[ks].sum()), "final_flagged": int(ff[ks].sum())}
    return {
        "n": len(ids), "positives": int(y.sum()),
        "baseline": stats(y, fb) | {"blocked_benign": blocked_benign(base), "latency_ms": lat(base)},
        "final": stats(y, ff) | {"blocked_benign": blocked_benign(fin), "latency_ms": lat(fin)},
        "youden_diff": round(float(youden(y, ff) - youden(y, fb)), 4),
        "youden_diff_ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "mcnemar": {"baseline_right_final_wrong": b_only, "final_right_baseline_wrong": f_only,
                    "p_exact": round(mcnemar_exact(b_only, f_only), 6)},
        "final_warn_from_context_only": len(ctx_only),
        "per_category": per_cat,
        "final_false_negatives": [i for k, i in enumerate(ids) if y[k] == 1 and not ff[k]],
        "final_false_positives": [i for k, i in enumerate(ids) if y[k] == 0 and ff[k]],
        "fixed_by_final": [i for k, i in enumerate(ids) if cf[k] and not cb[k]],
        "broken_by_final": [i for k, i in enumerate(ids) if cb[k] and not cf[k]],
    }


def fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x:.3f}" if not pct else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bdir, fdir, out = Path(a.baseline), Path(a.final), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bm, fm = json.loads((bdir / "metrics.json").read_text()), json.loads((fdir / "metrics.json").read_text())
    res = {"baseline_meta": bm["meta"], "final_meta": fm["meta"], "sets": {}}
    for name in fm["sets"]:
        res["sets"][name] = compare(load(bdir, name), load(fdir, name))
    (out / "comparison.json").write_text(json.dumps(res, indent=2))

    lines = ["| Set | System | TP | FP | FN | TN | Precision | Recall | F1 | FPR | Youden | Blocked benign | Latency mean / p95 (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["sets"].items():
        for sysname in ("baseline", "final"):
            s = r[sysname]
            lines.append(f"| {name} | {sysname} | {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} | {fmt(s['precision'])} | "
                         f"{fmt(s['recall'])} | {fmt(s['f1'])} | {fmt(s['fpr'])} | {fmt(s['youden'])} | {s['blocked_benign']} | "
                         f"{s['latency_ms']['mean']:.0f} / {s['latency_ms']['p95']:.0f} |")
    lines += ["", "| Set | Youden diff (final - baseline) | paired bootstrap 95% CI | McNemar: fixed / broken | exact p | final WARNs from context only |",
              "|---|---|---|---|---|---|"]
    for name, r in res["sets"].items():
        m = r["mcnemar"]
        lines.append(f"| {name} | {r['youden_diff']:+.3f} | [{r['youden_diff_ci95'][0]:+.3f}, {r['youden_diff_ci95'][1]:+.3f}] | "
                     f"{m['final_right_baseline_wrong']} / {m['baseline_right_final_wrong']} | {m['p_exact']:.4g} | "
                     f"{r['final_warn_from_context_only']} |")
    for name, r in res["sets"].items():
        lines += ["", f"### {name}: per category (flagged = WARN or BLOCK)", "",
                  "| Split | Category | Kind | n | Baseline flagged | Final flagged |", "|---|---|---|---|---|---|"]
        for k, v in r["per_category"].items():
            sp, c, kind = k.split("|")
            lines.append(f"| {sp} | {c} | {kind} | {v['n']} | {v['baseline_flagged']} | {v['final_flagged']} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2 + 2 * len(res["sets"]) + 3 + len(res["sets"])]))


if __name__ == "__main__":
    main()

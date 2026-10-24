"""Evaluate any Engine 2 model directory on labelled data with the production inference code.

    python -m ppaguard.engine2.evaluate --model-dir artifacts/engine2/default --data eval/sets/validation.jsonl
    python -m ppaguard.engine2.evaluate --model-dir <protectai dir> --data eval/sets/validation.jsonl --out report.json

Thresholds come from the model card unless --warn/--block are given. This tool never changes a model card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from .runtime import Detector


def evaluate(model_dir: str | Path, data_file: str | Path, warn: float | None = None, block: float | None = None) -> dict:
    det = Detector(model_dir)
    recs, rep = D.load_records(data_file)
    p = np.array([det.probability(r.text)[0] for r in recs])
    y = np.array([r.label for r in recs])
    thr = {"warn": warn if warn is not None else det.card.thresholds["warn"],
           "block": block if block is not None else det.card.thresholds["block"], "argmax_0_5": 0.5}
    by = {}
    for name in sorted({r.source for r in recs}):
        idx = [i for i, r in enumerate(recs) if r.source == name]
        by[name] = {"n": len(idx), "injection_rows": int(y[idx].sum()),
                    "flagged_at_warn": float((p[idx] > thr["warn"]).mean()), "blocked": float((p[idx] > thr["block"]).mean())}
    return {"model": det.status(), "data": rep.as_dict(), "thresholds": thr, "metrics": M.summary(y, p, thr), "by_subset": by}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--warn", type=float)
    ap.add_argument("--block", type=float)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = evaluate(a.model_dir, a.data, a.warn, a.block)
    txt = json.dumps(res, indent=2, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    m = res["metrics"]
    print(f"n={m['n']} roc_auc={m['roc_auc']:.4f} ap={m['average_precision']:.4f}")
    for k, r in m["at_threshold"].items():
        print(f"  {k:>10} >{r['threshold']:.2f}: injection P={r['injection']['precision']} R={r['injection']['recall']} "
              f"FPR={r['false_positive_rate']}")


if __name__ == "__main__":
    main()

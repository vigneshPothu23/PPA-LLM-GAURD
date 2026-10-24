"""Compare Engine 2 model folders on the frozen dev and test sets with the production inference code.

For every model:
  * dev  - ROC-AUC / AP, metrics at the model card's thresholds, at 0.5, and at thresholds re-selected on dev with the
           FPR-bounded rule (WARN FPR <= 10 %, BLOCK FPR <= 1 %);
  * test - the same threshold sets, evaluated once; per-subset confusion matrices (BIPIA e-mail/table/code hidden
           injections, NotInject trigger-word prompts).
A model trained on the dev set's source data is marked dev-contaminated: its dev numbers and dev-selected thresholds
are not meaningful, and only its provider threshold is used on test.

    python scripts/compare_detectors.py --model NAME=DIR [--model NAME=DIR ...] [--contaminated NAME] --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppaguard.engine2 import Detector  # noqa: E402
from ppaguard.engine2 import data as D  # noqa: E402
from ppaguard.engine2 import metrics as M  # noqa: E402


def block(y, p, t):
    r = M.report_at(y, p, t)
    i = r["injection"]
    return {"threshold": t, "tp": r["confusion"]["tp"], "fp": r["confusion"]["fp"], "tn": r["confusion"]["tn"],
            "fn": r["confusion"]["fn"], "precision": i["precision"], "recall": i["recall"], "f1": i["f1"],
            "fpr": r["false_positive_rate"], "accuracy": r["accuracy"]}


def evaluate(y, p, src, thresholds: dict) -> dict:
    out = {"n": int(len(y)), "positives": int(y.sum()), "roc_auc": M.roc_auc(y, p), "average_precision": M.average_precision(y, p),
           "at": {k: block(y, p, t) for k, t in thresholds.items()}, "subsets": {}}
    for s in sorted(set(src)):
        m = src == s
        out["subsets"][s] = {"n": int(m.sum()), "positives": int(y[m].sum()),
                             "roc_auc": M.roc_auc(y[m], p[m]) if 0 < y[m].sum() < m.sum() else None,
                             "at": {k: block(y[m], p[m], t) for k, t in thresholds.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--contaminated", action="append", default=[], help="models trained on the dev set's source")
    ap.add_argument("--dev", default=str(ROOT / "eval/sets/validation.jsonl"))
    ap.add_argument("--test", default=str(ROOT / "eval/sets/test.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, dev_rep = D.load_records(a.dev)
    test, test_rep = D.load_records(a.test)
    yd, yt = np.array([r.label for r in dev]), np.array([r.label for r in test])
    sd, st = np.array([r.source for r in dev]), np.array([r.source for r in test])
    report = {"dev_sha256": dev_rep.file_sha256, "test_sha256": test_rep.file_sha256, "models": {}}
    for spec in a.model:
        name, d = spec.split("=", 1)
        det = Detector(d, verify_hashes=not a.no_verify)
        t0 = time.perf_counter()
        pd_ = np.array([det.probability(r.text)[0] for r in dev])
        pt_ = np.array([det.probability(r.text)[0] for r in test])
        secs = time.perf_counter() - t0
        card_t = {"card_warn": det.card.thresholds["warn"], "card_block": det.card.thresholds["block"], "p_gt_0.5": 0.5}
        contaminated = name in a.contaminated
        sel = None if contaminated else M.select_thresholds(yd, pd_)
        thr = dict(card_t)
        if sel:
            thr.update(dev_warn=sel["warn"], dev_block=sel["block"])
        report["models"][name] = {
            "model_id": det.model_id, "provenance": det.card.provenance, "dev_contaminated": contaminated,
            "dev_selected_thresholds": sel, "seconds_for_dev_and_test": round(secs, 1),
            "ms_per_text": round(1000 * secs / (len(dev) + len(test)), 1),
            "dev": evaluate(yd, pd_, sd, thr), "test": evaluate(yt, pt_, st, thr)}
        with open(out / f"scores_{name}.csv", "w", newline="") as f:      # scores only, no text
            w = csv.writer(f)
            w.writerow(["split", "index", "source", "label", "p_injection"])
            w.writerows([("dev", i, r.source, r.label, round(float(x), 6)) for i, (r, x) in enumerate(zip(dev, pd_))])
            w.writerows([("test", i, r.source, r.label, round(float(x), 6)) for i, (r, x) in enumerate(zip(test, pt_))])
        print(name, "dev AUC", round(report["models"][name]["dev"]["roc_auc"], 4), "test AUC",
              round(report["models"][name]["test"]["roc_auc"], 4), f"{secs:.0f}s", flush=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

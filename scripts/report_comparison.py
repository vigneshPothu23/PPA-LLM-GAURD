"""Render one or more comparison.json files (from compare_detectors.py) as Markdown tables. Numbers are copied from the
JSON, never typed by hand.

    python scripts/report_comparison.py models/analysis/*/comparison.json > docs/v5/evidence/engine2/comparison.md
"""

import json
import sys


def f(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def main(paths):
    models = {}
    for p in paths:
        models.update(json.load(open(p))["models"])
    print("## Threshold-free ranking quality\n")
    print("| Model | Provenance | Dev ROC-AUC | Dev AP | Test ROC-AUC | Test AP | CPU ms/text |")
    print("|---|---|---|---|---|---|---|")
    for n, m in models.items():
        dv = "contaminated" if m["dev_contaminated"] else f(m["dev"]["roc_auc"])
        da = "contaminated" if m["dev_contaminated"] else f(m["dev"]["average_precision"])
        print(f"| {n} | {m['provenance']} | {dv} | {da} | {f(m['test']['roc_auc'])} | {f(m['test']['average_precision'])} | {m['ms_per_text']} |")
    print("\n## Injection class at operating thresholds (frozen test set, evaluated once)\n")
    print("Thresholds: `dev_*` re-selected on the dev set with the FPR-bounded rule; `card_*` from the model card; "
          "`p_gt_0.5` the model's argmax.\n")
    print("| Model | Threshold set | t | Precision | Recall | F1 | FPR | TP | FP | TN | FN |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, m in models.items():
        for k, b in m["test"]["at"].items():
            if k.startswith("card_") and ("dev_warn" in m["test"]["at"]):
                continue
            print(f"| {n} | {k} | {b['threshold']:.2f} | {f(b['precision'])} | {f(b['recall'])} | {f(b['f1'])} | {f(b['fpr'], True)} "
                  f"| {b['tp']} | {b['fp']} | {b['tn']} | {b['fn']} |")
    print("\n## Per-subset behaviour on the test set (WARN-level threshold)\n")
    print("| Model | Subset | n | Injections caught | Benign flagged | Subset ROC-AUC |")
    print("|---|---|---|---|---|---|")
    for n, m in models.items():
        key = "dev_warn" if "dev_warn" in m["test"]["at"] else "card_warn"
        for s, v in m["test"]["subsets"].items():
            b = v["at"][key]
            pos, neg = v["positives"], v["n"] - v["positives"]
            print(f"| {n} | {s} | {v['n']} | {b['tp']}/{pos} | {b['fp']}/{neg} | {f(v['roc_auc'])} |")
    print("\n## Dev set (thresholds chosen here)\n")
    print("| Model | Threshold set | t | Precision | Recall | F1 | FPR |")
    print("|---|---|---|---|---|---|---|")
    for n, m in models.items():
        if m["dev_contaminated"]:
            print(f"| {n} | – | – | trained on the dev set's source | | | |")
            continue
        for k in ("dev_warn", "dev_block", "p_gt_0.5"):
            b = m["dev"]["at"].get(k)
            if b:
                print(f"| {n} | {k} | {b['threshold']:.2f} | {f(b['precision'])} | {f(b['recall'])} | {f(b['f1'])} | {f(b['fpr'], True)} |")


if __name__ == "__main__":
    main(sys.argv[1:])

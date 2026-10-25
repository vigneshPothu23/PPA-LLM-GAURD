"""End-to-end evaluation through the REAL application workflow (not a detector-only benchmark).

Every input is submitted with `SecurityWorkflow.submit()` exactly as the Scan Console does: a demo organization, an
administrator, an application with the default "general" profile, the configured Engine 2 model, the policy engine,
Engine 1 assembly and the audit record. The decision returned by the application is what is scored.

    python eval/run_pipeline_eval.py --code <checkout> --set scenarios=eval/scenarios/scenarios.jsonl \
        --set frozen_test=eval/sets/test.jsonl --out <dir>

`--code` is the repository checkout whose `src/` is imported, so the same harness evaluates the baseline (e.g. a
worktree of `main`) and the modified code with the identical procedure and inputs.

Scoring: a row is FLAGGED when the application decision is WARN or BLOCK (nothing is forwarded without review), and
ALLOWED otherwise. Positives are label-1 rows. Outputs: per-row predictions (text-free), metrics.json, metrics.md.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                r = json.loads(line)
                r.setdefault("id", f"row{i}")
                r["category"] = r.get("category") or r.get("set") or r.get("group") or "all"
                r["split"] = r.get("split") or "all"
                rows.append(r)
    return rows


def counts(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["label"] == 1 and r["flagged"])
    fn = sum(1 for r in rows if r["label"] == 1 and not r["flagged"])
    fp = sum(1 for r in rows if r["label"] == 0 and r["flagged"])
    tn = sum(1 for r in rows if r["label"] == 0 and not r["flagged"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    fpr = fp / (fp + tn) if fp + tn else None
    blocked_benign = sum(1 for r in rows if r["label"] == 0 and r["decision"] == "BLOCK")
    blocked_attack = sum(1 for r in rows if r["label"] == 1 and r["decision"] == "BLOCK")
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "blocked_attack": blocked_attack, "blocked_benign": blocked_benign}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="repository checkout to evaluate (its src/ is imported)")
    ap.add_argument("--set", action="append", required=True, help="NAME=PATH (JSONL with text, label)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="free-text name of the configuration being evaluated")
    ap.add_argument("--threads", default="2")
    a = ap.parse_args()

    code = Path(a.code).resolve()
    sys.path.insert(0, str(code / "src"))
    os.environ.setdefault("PPAGUARD_THREADS", a.threads)
    from ppaguard.core import db as dbm
    from ppaguard.core.authz import load_principal
    from ppaguard.core.runtime import load_detector
    from ppaguard.core.services import Services, seed_demo
    from ppaguard.core.workflow import SecurityWorkflow

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ppaguard-eval-"))
    engine = dbm.make_engine(f"sqlite:///{tmp / 'eval.db'}")
    seed_demo(engine)
    svc = Services(engine)
    with engine.connect() as c:
        admin = load_principal(c, "admin@acme.example")
    # The application rate-limits each user per application (max 600/min). The harness does not change that control;
    # it spreads rows over ten identically configured applications so evaluation speed never hits it.
    app_ids = []
    for k in range(10):
        app_ids.append(svc.create_app(admin, f"eval-harness-{k}", "documents", "general"))
        svc.update_policy(admin, app_ids[-1], {"rate_limit_per_min": 600})
    detector, status = load_detector()
    if not status.loaded:
        raise SystemExit(f"Engine 2 model not loaded: {status.error}")
    wf = SecurityWorkflow(engine, detector, status.error)
    meta = {"code": str(code), "label": a.label, "model": status.model_id, "provenance": status.provenance,
            "thresholds": dict(detector.card.thresholds), "app_profile": "general"}
    print("evaluating", meta, flush=True)

    report = {"meta": meta, "sets": {}}
    for spec in a.set:
        name, path = spec.split("=", 1)
        rows = load_rows(Path(path))
        preds, lat = [], []
        t_set = time.perf_counter()
        for i, r in enumerate(rows):
            res = wf.submit(admin, app_ids[i % len(app_ids)], r["text"], channel="eval")
            e2 = res.engine2 or {}
            strong = [f for f in (res.engine1 or {}).get("forgery_findings", []) if f.get("strong")]
            preds.append({"id": r["id"], "split": r["split"], "category": r["category"], "label": int(r["label"]),
                          "decision": res.decision.name, "outcome": res.outcome, "flagged": res.decision.name != "ALLOW",
                          "probability": e2.get("probability"), "detector_band": (res.policy or {}).get("detector_band"),
                          "warn_threshold": (res.policy or {}).get("warn_threshold"),
                          "block_threshold": (res.policy or {}).get("block_threshold"),
                          "engine1_wrapped": bool((res.engine1 or {}).get("applied")), "forgery_strong": len(strong),
                          "reasons": res.reasons, "latency_ms": round(res.latency_ms, 1),
                          "extra": {k: v for k, v in e2.items() if k.startswith("segment") or k == "context"}})
            lat.append(res.latency_ms)
        by_split, by_cat = defaultdict(list), defaultdict(list)
        for p in preds:
            by_split[p["split"]].append(p)
            by_cat[(p["split"], p["category"], p["label"])].append(p)
        report["sets"][name] = {
            "overall": counts(preds),
            "splits": {s: counts(v) for s, v in sorted(by_split.items())},
            "categories": {f"{s}|{c}|{'attack' if l else 'benign'}": counts(v) for (s, c, l), v in sorted(by_cat.items())},
            "latency_ms": {"mean": round(statistics.mean(lat), 1),
                           "p95": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 1), "n": len(lat)},
            "seconds": round(time.perf_counter() - t_set, 1),
            "forwarded_without_review_attacks": sum(1 for p in preds if p["label"] == 1 and p["outcome"] == "protected_prompt_returned"),
            "blocked_but_wrapped": sum(1 for p in preds if p["decision"] == "BLOCK" and p["engine1_wrapped"]),
        }
        with open(out / f"predictions_{name}.jsonl", "w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p) + "\n")
        o = report["sets"][name]["overall"]
        print(f"{name}: n={o['n']} TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']} recall={o['recall']} fpr={o['fpr']} "
              f"latency mean={report['sets'][name]['latency_ms']['mean']}ms", flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

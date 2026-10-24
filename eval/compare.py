"""Engine 2 evaluation: pretrained ProtectAI DeBERTa (ONNX, as shipped in the app) vs the frozen
TF-IDF + LR baseline, on identical frozen sets. Evaluation only: nothing is trained.

Operating points (all fixed BEFORE the test set is scored):
  DeBERTa  official   : 2-decimal score  > 0.92   (LLM Guard default)
  DeBERTa  argmax     : 2-decimal score  > 0.50   (model card pipeline)
  DeBERTa  val-tuned  : raw score       >= t, t chosen on validation.jsonl only
                        (project rule: best F1 with FPR <= 5 %)
  baseline WARN/BLOCK : its frozen thresholds from model_card.json (raw score >= t)

Usage:
  python eval/compare.py --sets eval/sets --model-dir <folder with onnx/> \
      [--baseline-src <protect-llm-agent-v2/src> --baseline-dir <.../models/baseline>] --out eval/results
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppa_guard.engine2 import HF_REVISION, DebertaDetector  # noqa: E402


def load(sets, name):
    return [json.loads(l) for l in open(Path(sets) / f"{name}.jsonl", encoding="utf-8")]


def counts(y, pred):
    y = np.asarray(y); p = np.asarray(pred, bool)
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if rec is not None else None)
    r = lambda v: None if v is None else round(v, 4)  # noqa: E731
    return {"n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "fpr": r(fp / (fp + tn) if fp + tn else None),
            "recall": r(rec), "precision": r(prec), "f1": r(f1)}


def breakdown(rows, pred):
    y = [r["label"] for r in rows]
    out = {"overall": counts(y, pred)}
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["set"], []).append(i)
        if r["set"].startswith("bipia"):
            groups.setdefault("bipia_all", []).append(i)
        if r["set"] in ("notinject_two", "notinject_three"):
            groups.setdefault("notinject_test", []).append(i)
    for g, idx in groups.items():
        out[g] = counts([y[i] for i in idx], [pred[i] for i in idx])
    return out


def select_threshold(y, s, max_fpr=0.05):
    """Validation-only: best F1 among thresholds with FPR <= max_fpr (same rule as the project)."""
    y = np.asarray(y); s = np.asarray(s); best = None
    for t in np.unique(np.round(s, 6)):
        c = counts(y, s >= t)
        if c["fpr"] is not None and c["fpr"] <= max_fpr:
            key = (c["f1"] or 0, t)
            if best is None or key > best[0]:
                best = (key, float(t))
    return best[1] if best else 1.0


def timed(fn, rows):
    s, t = [], []
    for r in rows:
        t0 = time.perf_counter(); s.append(fn(r["text"])); t.append((time.perf_counter() - t0) * 1000)
    return np.array(s), np.array(t)


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=str(ROOT / "eval/sets"))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--baseline-src"); ap.add_argument("--baseline-dir")
    ap.add_argument("--out", default=str(ROOT / "eval/results"))
    a = ap.parse_args()
    val, test = load(a.sets, "validation"), load(a.sets, "test")
    yv = [r["label"] for r in val]
    res = {"env": {"python": platform.python_version(), "machine": platform.machine(), "threads": a.threads},
           "sets": {"validation": len(val), "test": len(test)}}

    m0 = rss_mb(); t0 = time.perf_counter()
    det = DebertaDetector(a.model_dir, revision=HF_REVISION, threads=a.threads)
    load_s = time.perf_counter() - t0; m1 = rss_mb()
    info = {}
    def dscore(text):
        d = det.detect(text); info.setdefault("trunc", []).append(d.truncated); return d.raw_score
    sv, _ = timed(dscore, val); trunc_val = sum(info.pop("trunc"))
    t_val = select_threshold(yv, sv)
    st, lat = timed(dscore, test); trunc_test = sum(info.pop("trunc"))
    rv, rt = np.round(sv, 2), np.round(st, 2)
    res["deberta"] = {
        "model_id": det.model_id, "load_seconds": round(load_s, 1),
        "rss_mb_after_load": None if m1 is None else round(m1), "rss_mb_load_increase": None if m0 is None else round(m1 - m0),
        "rss_mb_after_eval": None if m0 is None else round(rss_mb()),
        "val_tuned_threshold": round(t_val, 4), "truncated_inputs": {"validation": trunc_val, "test": trunc_test},
        "latency_ms": {"mean": round(float(lat.mean()), 1), "p50": round(float(np.percentile(lat, 50)), 1),
                       "p95": round(float(np.percentile(lat, 95)), 1)},
        "validation": {"official>0.92": breakdown(val, rv > 0.92), "argmax>0.50": breakdown(val, rv > 0.5),
                       f"val-tuned>={t_val:.4f}": breakdown(val, sv >= t_val)},
        "test": {"official>0.92": breakdown(test, rt > 0.92), "argmax>0.50": breakdown(test, rt > 0.5),
                 f"val-tuned>={t_val:.4f}": breakdown(test, st >= t_val)},
    }
    if a.baseline_src and a.baseline_dir:
        sys.path.insert(0, a.baseline_src)
        from protect_llm.engine2.baseline import BaselineDetector
        b = BaselineDetector.load(a.baseline_dir)
        bv, _ = timed(lambda t: b.detect(t).score, val); bt, blat = timed(lambda t: b.detect(t).score, test)
        res["baseline"] = {"model_id": b.model_id,
                           "latency_ms": {"mean": round(float(blat.mean()), 1), "p95": round(float(np.percentile(blat, 95)), 1)},
                           "validation": {f"WARN>={b.warn_threshold:.4f}": breakdown(val, bv >= b.warn_threshold),
                                          f"BLOCK>={b.block_threshold:.4f}": breakdown(val, bv >= b.block_threshold)},
                           "test": {f"WARN>={b.warn_threshold:.4f}": breakdown(test, bt >= b.warn_threshold),
                                    f"BLOCK>={b.block_threshold:.4f}": breakdown(test, bt >= b.block_threshold)}}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(res, indent=1))

    def fmt(c, k):
        v = c.get(k); return "  n/a " if v is None else f"{v:6.3f}"
    print(f"{'model / operating point':34s} | NotInject FPR | BIPIA recall  FPR    F1 | deepset(val) recall  FPR    F1")
    for m in ("deberta", "baseline"):
        if m not in res:
            continue
        for p in res[m]["test"]:
            t, v = res[m]["test"][p], res[m]["validation"][p]
            print(f"{m + ' ' + p:34s} | {fmt(t['notinject_test'], 'fpr'):>13} | {fmt(t['bipia_all'], 'recall')} "
                  f"{fmt(t['bipia_all'], 'fpr')} {fmt(t['bipia_all'], 'f1')} | {fmt(v['deepset_train'], 'recall')} "
                  f"{fmt(v['deepset_train'], 'fpr')} {fmt(v['deepset_train'], 'f1')}")
    d = res["deberta"]
    print(f"DeBERTa latency mean {d['latency_ms']['mean']} ms, p95 {d['latency_ms']['p95']} ms; load {d['load_seconds']} s; "
          f"RSS after load {d['rss_mb_after_load']} MB, after eval {d['rss_mb_after_eval']} MB; truncated {d['truncated_inputs']}")
    if "baseline" in res:
        print(f"baseline latency mean {res['baseline']['latency_ms']['mean']} ms")
    print("wrote", out / "comparison.json")


if __name__ == "__main__":
    main()

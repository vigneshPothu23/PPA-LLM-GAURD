"""Export the evaluated guard_full fusion model as an artifact and check live/cached parity.

  * Fits guard_full on the ctx_dev cache with exactly the code used for the evaluation
    (evaluate_from_cache.fit_ablation), and writes artifacts/context_guard/guard_full.json with the
    thresholds selected on ctx_dev, provenance and status EXPERIMENTAL.
  * Parity: loads the artifact with the real detector + embedder and scores rows of unseen_synth live;
    the live document probabilities must equal the cached-path probabilities (max abs diff <= 1e-4; the cache rounds features to 6 decimals),
    so the evaluation numbers describe the shipped component.

    python scripts/context_guard/export_guard_artifact.py [--parity-rows 25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_from_cache as E  # noqa: E402
from paths import detector_dir, minilm_dir  # noqa: E402
from ppaguard.engine2 import Detector  # noqa: E402
from ppaguard.experimental.context_guard import ContextGuard  # noqa: E402
from ppaguard.experimental.context_guard.embed import Embedder  # noqa: E402

DEST = ROOT / "artifacts/context_guard/guard_full.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None, help="Engine 2 model folder (default: the app's ProtectAI folder)")
    ap.add_argument("--minilm", default=None, help="MiniLM folder (default: downloaded on first use)")
    ap.add_argument("--parity-rows", type=int, default=25)
    a = ap.parse_args()
    a.detector, a.minilm = detector_dir(a.detector), minilm_dir(a.minilm)

    ctx = E.load("ctx_dev")
    model = E.fit_ablation(ctx, ())
    comp = json.loads((E.OUT / "comparison.json").read_text())
    th = comp["thresholds"]["guard_full"]
    det_card = json.loads((Path(a.detector) / "model_card.json").read_text())
    art = {
        "name": "context-guard-full",
        "status": "EXPERIMENTAL - not validated for production; see docs/v5/CONTEXT_GUARD.md",
        "features": E.FEATURES,
        "use_offtopic": True, "use_detector": True,
        "max_detector_calls": ContextGuard.MAX_DETECTOR_CALLS,
        "model": {"feature_names": model.feature_names, "mean": model.mean, "std": model.std,
                  "weights": model.weights, "bias": model.bias},
        "thresholds_ctx_dev": {"WARN": th["WARN"], "BLOCK": th["BLOCK"]},
        "policy_bridge": "guard alone never BLOCKs by default (BLOCK precision target not met on ctx_dev)",
        "fitted_on": {"set": "eval/context_sets/ctx_dev.jsonl", "sha256": sha(ROOT / "eval/context_sets/ctx_dev.jsonl"),
                      "rows": len(ctx), "supervision": "segment-level (known payload segments)"},
        "requires": {
            "detector": {"model_id": det_card.get("model_id"), "note": "ppaguard Engine 2 model folder (default ProtectAI v2)"},
            "embedder": {"model_id": "sentence-transformers/all-MiniLM-L6-v2",
                         "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "licence": "Apache-2.0",
                         "onnx_sha256": sha(Path(a.minilm) / "onnx/model.onnx")},
        },
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(art, indent=2))
    print("wrote", DEST)

    # ---- parity: live pipeline vs cached-feature pipeline ----
    det = Detector(a.detector, threads=2, verify_hashes=False)
    emb = Embedder(a.minilm, threads=2)
    g = ContextGuard.load(DEST, detector=det, embedder=emb)
    rows = [json.loads(l) for l in open(ROOT / "eval/context_sets/unseen_synth.jsonl")][: a.parity_rows]
    cached = {r["id"]: r for r in E.load("unseen_synth")}
    live = np.array([g.probability(r["text"]) for r in rows])
    cache_p = E.doc_scores(model, [cached[r["id"]] for r in rows], ())
    diff = float(np.abs(live - cache_p).max())
    print(f"parity on {len(rows)} unseen_synth rows: max |live - cached| = {diff:.2e}")
    # the cache stores features rounded to 6 decimals, so ~1e-6 differences are expected rounding
    if diff > 1e-4:
        raise SystemExit("PARITY FAILED")
    print("PARITY OK")


if __name__ == "__main__":
    main()

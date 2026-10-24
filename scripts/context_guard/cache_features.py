"""Precompute and cache per-document segment features for ONE evaluation set (resumable).

This is the only compute-heavy step. For each row it stores:
  * X: the full segment feature matrix (segments x 10 features, the ContextGuard.FEATURES order),
    computed once with detector + embedder enabled, so every guard ablation is just a column subset;
  * kinds: the segment kinds;
  * seg_pos: indices of segments that overlap the known payload (segment-level positive labels);
  * wholedoc: whole-document P(injection) for each baseline detector.

Resumable: rows already in the output cache (keyed by id) are skipped, and the file is flushed every row,
so re-running after an interruption continues where it stopped. Run one set per invocation to stay within
background time limits.

    python scripts/context_guard/cache_features.py --set frozen_test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import detector_dir, minilm_dir  # noqa: E402

from ppaguard.engine2 import Detector  # noqa: E402
from ppaguard.experimental.context_guard import ContextGuard  # noqa: E402
from ppaguard.experimental.context_guard.embed import Embedder  # noqa: E402

SETS = {
    "ctx_dev": ROOT / "eval/context_sets/ctx_dev.jsonl",
    "frozen_dev": ROOT / "eval/sets/validation.jsonl",
    "frozen_test": ROOT / "eval/sets/test.jsonl",
    "unseen_synth": ROOT / "eval/context_sets/unseen_synth.jsonl",
    "unseen_email": ROOT / "eval/context_sets/unseen_email.jsonl",
    "unseen_email_2": ROOT / "eval/context_sets/unseen_email_2.jsonl",  # confirmatory, scored once
    "dev2_email": ROOT / "eval/context_sets/dev2_email.jsonl",          # round-2 development
    "unseen_email_3": ROOT / "eval/context_sets/unseen_email_3.jsonl",  # round-2 test, scored once
    "unseen_embed_3": ROOT / "eval/context_sets/unseen_embed_3.jsonl",  # round-2 test, scored once
}
CACHE = ROOT / "docs/v5/evidence/context_guard/cache"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def overlaps(seg_text: str, payload: str) -> bool:
    s, p = norm(seg_text), norm(payload)
    if not s or not p:
        return False
    return s in p or p in s or (len(s) > 12 and s[:40] in p)


def row_id(r: dict, i: int) -> str:
    return r.get("id") or f"row{i}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--detector", default=None, help="Engine 2 model folder (default: the app's ProtectAI folder)")
    ap.add_argument("--minilm", default=None, help="MiniLM folder (default: downloaded on first use)")
    ap.add_argument("--baselines", default="roberta6l=artifacts/engine2/roberta-6l-experimental,"
                                           "scratch=artifacts/engine2/scratch-baseline")
    args = ap.parse_args()
    args.detector, args.minilm = detector_dir(args.detector), minilm_dir(args.minilm)

    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / f"{args.set}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["id"])

    det = Detector(args.detector, threads=args.threads, verify_hashes=False)
    emb = Embedder(args.minilm, threads=args.threads)
    guard = ContextGuard(detector=det, embedder=emb, use_offtopic=True, use_detector=True)

    baselines = {}
    for spec in args.baselines.split(","):
        if spec.strip():
            nm, d = spec.split("=", 1)
            baselines[nm] = Detector(str(ROOT / d) if not Path(d).is_absolute() else d,
                                     threads=args.threads, verify_hashes=False)

    with open(SETS[args.set]) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]

    n_new = 0
    with open(out_path, "a") as out:
        for i, r in enumerate(rows):
            rid = row_id(r, i)
            if rid in done:
                continue
            text = r["text"]
            segs, X, _, _, _ = guard.segment_matrix(text)
            payload = r.get("payload", "")
            seg_pos = [j for j, s in enumerate(segs) if payload and overlaps(s.text, payload)]
            wholedoc = {"protectai": float(det.probability(text)[0])}
            for nm, d in baselines.items():
                wholedoc[nm] = float(d.probability(text)[0])
            rec = {"id": rid, "label": int(r["label"]),
                   "group": r.get("group") or r.get("set") or "all",
                   "kinds": [s.kind for s in segs],
                   "X": np.asarray(X, dtype=np.float64).round(6).tolist(),
                   "seg_pos": seg_pos, "wholedoc": wholedoc}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_new += 1
            if n_new % 50 == 0:
                print(f"{args.set}: {n_new} new ({i + 1}/{len(rows)})", flush=True)
    print(f"{args.set}: done, {n_new} new rows, total {len(done) + n_new}", flush=True)


if __name__ == "__main__":
    main()

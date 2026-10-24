"""Add whole-document scores of extra baseline detectors to an existing feature cache (resumable).

    python scripts/context_guard/add_baseline_scores.py --set frozen_test \
        --model deepset=<folder> --model fmops=<folder>   (folders made by scripts/prepare_hf_classifier.py)

Rows that already carry a score for a model are skipped. The cache is rewritten atomically (temp file +
rename) every 100 rows, so an interruption loses at most the last batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_features import CACHE, SETS  # noqa: E402
from ppaguard.engine2 import Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--model", action="append", required=True, help="NAME=DIR")
    a = ap.parse_args()
    path = CACHE / f"{a.set}.jsonl"
    with open(path) as fh:
        cache = [json.loads(l) for l in fh if l.strip()]
    with open(SETS[a.set]) as fh:
        src = [json.loads(l) for l in fh if l.strip()]
    if len(cache) != len(src):
        raise SystemExit(f"{a.set}: cache incomplete ({len(cache)}/{len(src)}); finish cache_features first")
    dets = {n: Detector(d, threads=2, verify_hashes=False) for n, d in (s.split("=", 1) for s in a.model)}

    def save():
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in cache:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    done = 0
    for r, s in zip(cache, src):
        assert r["label"] == int(s["label"])
        todo = [n for n in dets if n not in r["wholedoc"]]
        for n in todo:
            r["wholedoc"][n] = float(dets[n].probability(s["text"])[0])
        if todo:
            done += 1
            if done % 100 == 0:
                save()
                print(f"{a.set}: {done} rows scored", flush=True)
    save()
    print(f"{a.set}: done ({done} rows updated)", flush=True)


if __name__ == "__main__":
    main()

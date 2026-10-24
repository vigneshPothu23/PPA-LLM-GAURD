"""Build the corrected indirect-injection development set (dev_indirect_v2.jsonl).

Fixes to dev_indirect.jsonl (44,250 rows):
  * holdout is per UNIQUE context (the v1 file used 180 table rows = 163 unique tables -> 3,825 duplicate rows)
  * every context that also appears in ANY BIPIA test split is excluded (v1 had 2 such emails)
  * one CLEAN row (label 0) per context -> false-positive rate becomes measurable
  * exactly one poisoned row (label 1) per context x position, attack chosen deterministically and spread
    across attack categories (v1 crossed every context with every attack: 91.5 % tables)
  * integer labels (v1 stored "1" as a string)

Only BIPIA TRAIN contexts and TRAIN attack pools are used. The frozen test set is never read.
Contexts: email and code use ALL BIPIA-train contexts not shared with BIPIA-test (34 and 50; v1 had only
10 each, too few for usable confidence intervals); table uses the 163 unique tables held out in v1.
Every context used here must be excluded from any later fine-tuning data. The v1 file is read, never modified.

Usage:
  python experiments/scripts/build_dev_v2.py --bipia <BIPIA repo> --v1 experiments/data/dev_indirect.jsonl \
      --out experiments/data/dev_indirect_v2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bipia", required=True)
ap.add_argument("--v1", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)  # v1 middle insertions reproduce with seed 42
ap.add_argument("--all-eligible", default="email,code",
                help="tasks for which ALL BIPIA-train contexts not shared with BIPIA-test are used (more power); "
                     "other tasks use the v1 held-out contexts")
args = ap.parse_args()

sys.path.insert(0, args.bipia)
from bipia.data.utils import insert_end, insert_middle, insert_start  # noqa: E402  (official BIPIA code)

B = Path(args.bipia) / "benchmark"
POSITIONS = (("end", insert_end), ("start", insert_start), ("middle", insert_middle))
POOL_FILE = {"email": "text_attack_train.json", "table": "text_attack_train.json", "code": "code_attack_train.json"}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contexts(task: str, split: str) -> list[str]:
    rows = [json.loads(l) for l in open(B / task / f"{split}.jsonl", encoding="utf-8")]
    return ["\n".join(r["context"]) if task == "code" else r["context"] for r in rows]


# ---- recover the v1 held-out contexts (reverse the official insertion) ----
v1 = [json.loads(l) for l in open(args.v1, encoding="utf-8") if l.strip()]
v1_ctx: dict[str, set[str]] = collections.defaultdict(set)
for r in v1:
    t, a, p = r["text"], r["attack"], r["position"]
    if p == "end":
        c = t[: -len(a) - 1]
    elif p == "start":
        c = t[len(a) + 1:]
    else:
        i = t.index("\n" + a + "\n")
        c = t[:i] + t[i + len(a) + 2:]
    v1_ctx[r["task"]].add(c)

rows, log = [], {"v1_file_sha256": file_sha(args.v1), "seed": args.seed, "tasks": {}}
for task in ("email", "table", "code"):
    train, test = set(contexts(task, "train")), set(contexts(task, "test"))
    chosen = sorted(train if task in args.all_eligible.split(",") else v1_ctx[task])
    assert set(chosen) <= train, f"{task}: v1 context not in BIPIA train"
    leaked = [c for c in chosen if c in test]
    kept = [c for c in chosen if c not in test]
    raw = json.load(open(B / POOL_FILE[task], encoding="utf-8"))
    by_cat = [(cat, s) for cat in sorted(raw) for s in raw[cat]]        # deterministic order
    test_pool = {s for f in ("text_attack_test.json", "code_attack_test.json")
                 for v in json.load(open(B / f, encoding="utf-8")).values() for s in v}
    assert not any(s in test_pool for _, s in by_cat), "train pool overlaps test pool"
    k = 0
    for c in kept:
        cid = sha(c)[:16]
        rows.append({"id": f"{task}-{cid}-clean", "task": task, "position": "clean", "label": 0,
                     "context_id": cid, "attack": None, "attack_category": None, "text": c})
        for pos, fn in POSITIONS:
            cat, att = by_cat[(int(cid, 16) + k) % len(by_cat)]       # rotate through categories
            k += 7                                                    # stride co-prime with pool size
            rows.append({"id": f"{task}-{cid}-{pos}", "task": task, "position": pos, "label": 1,
                         "context_id": cid, "attack": att, "attack_category": cat,
                         "text": fn(c, att, random_state=args.seed)})
    log["tasks"][task] = {"context_source": "all eligible BIPIA train" if task in args.all_eligible.split(",") else "v1 holdout",
                          "candidate_unique_contexts": len(chosen), "excluded_shared_with_bipia_test": len(leaked),
                          "kept_contexts": len(kept), "clean_rows": len(kept), "poisoned_rows": 3 * len(kept),
                          "attack_categories_used": len({r["attack_category"] for r in rows if r["task"] == task and r["label"]})}

texts = [r["text"] for r in rows]
assert len(texts) == len(set(texts)), "duplicate texts in v2"
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
log.update({"rows": len(rows), "labels": dict(collections.Counter(r["label"] for r in rows)),
            "duplicates": 0, "out_sha256": file_sha(out)})
Path(str(out) + ".build_log.json").write_text(json.dumps(log, indent=1))
print(json.dumps(log, indent=1))

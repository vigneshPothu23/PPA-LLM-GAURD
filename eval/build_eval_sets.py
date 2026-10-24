"""Build the frozen comparison sets (read-only on all sources).

validation.jsonl  (threshold selection only)
    deepset/prompt-injections train split, 546 rows  (copy shipped in the InjecGuard repo)
    NotInject_one, 113 benign rows
test.jsonl  (scored once)
    NotInject_two + NotInject_three, 226 benign rows
    BIPIA indirect injection, built with BIPIA's OWN builders from its TEST contexts and TEST attack pools:
      email (50 contexts), table (100), code (50)
      label 0: each clean context
      label 1: for each context and each official position (end/start/middle), ONE attack drawn
               deterministically from the test pool  -> 3 poisoned rows per context
Missing (not reachable from the build machine): deepset test split (116 rows, Hugging Face only).

Usage:
  python build_eval_sets.py --repos <folder with InjecGuard and BIPIA clones> --out sets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repos", required=True)
ap.add_argument("--out", default="sets")
ap.add_argument("--seed", type=int, default=2023)  # BIPIA README uses seed 2023
args = ap.parse_args()
repos = Path(args.repos)
sys.path.insert(0, str(repos / "BIPIA"))
from bipia.data import CodeIPIABuilder, EmailIPIABuilder, TableIPIABuilder  # noqa: E402
from bipia.data.utils import insert_end, insert_middle, insert_start  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def rows_from_json(path):
    return json.load(open(path, encoding="utf-8"))


# ---------------- validation ----------------
val = []
pg = rows_from_json(repos / "InjecGuard/datasets/train.json")
for r in pg:
    if r["source"] == "prompt-injections":
        val.append({"text": r["prompt"], "label": int(r["label"]), "set": "deepset_train", "group": "deepset"})
for r in rows_from_json(repos / "InjecGuard/datasets/NotInject_one.json"):
    val.append({"text": r["prompt"], "label": 0, "set": "notinject_one", "group": r["category"]})

# ---------------- test ----------------
test = []
for k in ("two", "three"):
    for r in rows_from_json(repos / f"InjecGuard/datasets/NotInject_{k}.json"):
        test.append({"text": r["prompt"], "label": 0, "set": f"notinject_{k}", "group": r["category"]})

bench = repos / "BIPIA/benchmark"
tasks = [("email", EmailIPIABuilder, "text_attack_test.json"),
         ("table", TableIPIABuilder, "text_attack_test.json"),
         ("code", CodeIPIABuilder, "code_attack_test.json")]
positions = [("end", insert_end), ("start", insert_start), ("middle", insert_middle)]
rng = random.Random(args.seed)
for task, cls, attack_file in tasks:
    builder = cls(seed=args.seed)
    builder.enable_stealth = False  # BIPIA default (plain-text attacks); set by __call__ normally
    contexts = builder.load_context(str(bench / task / "test.jsonl"))
    attacks = builder.load_attack(str(bench / attack_file))
    names = sorted(attacks)
    for ctx in contexts:
        clean = "\n".join(ctx["context"]) if task == "code" else ctx["context"]
        test.append({"text": clean, "label": 0, "set": f"bipia_{task}", "group": "clean"})
    for pos_name, fn in positions:
        # one official builder call per (context, attack) would be 50-100x larger; we sample one attack per context
        for ctx in contexts:
            name = rng.choice(names)
            clean = "\n".join(ctx["context"]) if task == "code" else ctx["context"]
            poisoned = fn(clean, attacks[name], random_state=args.seed)
            test.append({"text": poisoned, "label": 1, "set": f"bipia_{task}", "group": f"{pos_name}:{name.rsplit('-', 1)[0]}"})

for name, rows in (("validation", val), ("test", test)):
    p = out / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    by = {}
    for r in rows:
        by.setdefault(r["set"], [0, 0])[r["label"]] += 1
    print(f"{p}  rows={len(rows)}  sha256={h}")
    for s, (n0, n1) in by.items():
        print(f"   {s:16s} benign={n0:4d}  malicious={n1:4d}")
print("MISSING: deepset test split (116 rows) - only on Hugging Face, not reachable from this machine")

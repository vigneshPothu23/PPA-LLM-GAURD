"""Final held-out set for the submission evaluation (built 2026-10-25 BEFORE any design decision of that phase, and
scored exactly once after the final configuration was pre-registered).

final_holdout.jsonl (real e-mail text; git-ignored like every other context set)
  * 250 LLMail-Inject PHASE-2 attack e-mails not used in unseen_email_2 / unseen_email_3          label 1, llmail_attack_f
  * 250 Enron e-mails from a NEW byte range (enron5): 125 containing a trigger word, 125 without  label 0, enron_trigger_f / enron_plain_f
  * 150 other enron5 e-mails with an embedded published injection:
      100 held-out TaskTracker segments not used before + 50 short phase-2 LLMail bodies           label 1, embed_tasktracker_f / embed_llmail_f
  * 150 other enron5 e-mails, unmodified                                                          label 0, enron_clean_f
Every text is deduplicated (normalised) against the frozen sets and every earlier context set. No attack text is
authored here. Requires `download_sources.py` (enron5 range) and the audited training candidate (TaskTracker source).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_datasets as BD  # noqa: E402
from build_round2_sets import OUT, TRIGGER, Pool, enron, llmail_bodies, norm, read_jsonl, row  # noqa: E402
from paths import SOURCES  # noqa: E402

ROOT = HERE.parents[1]
SEED = 20261025_2


def main() -> None:
    rng = random.Random(SEED)
    earlier = [ROOT / "eval/sets/validation.jsonl", ROOT / "eval/sets/test.jsonl"] + \
              [OUT / f"{n}.jsonl" for n in ("ctx_dev", "unseen_synth", "unseen_email", "unseen_email_2", "dev2_email",
                                             "unseen_email_3", "unseen_embed_3")]
    seen = {norm(r["text"]) for p in earlier for r in read_jsonl(p)}
    n_earlier = len(seen)
    pool = Pool(seen)

    all_inj = BD.recover_tasktracker_injections()
    odd = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2  # noqa: E731
    used = {norm(r["payload"]) for n in ("ctx_dev", "unseen_synth", "dev2_email", "unseen_embed_3")
            for r in read_jsonl(OUT / f"{n}.jsonl") if r.get("payload")}
    held_out = sorted(s for s in all_inj if odd(s) == 1 and norm(s) not in used)
    rng.shuffle(held_out)

    p2 = llmail_bodies(SOURCES / "llmail2" / "labelled_unique_submissions_phase2.json")
    rng.shuffle(p2)
    e5 = enron("enron5")
    rng.shuffle(e5)
    short = lambda t: 40 <= len(t) <= 400  # noqa: E731

    rows = [row(t, 1, "final_holdout", "llmail_attack_f", "llmail_phase2") for t in pool.fresh(p2, 250)]
    rows += [row(t, 0, "final_holdout", "enron_trigger_f", "none") for t in pool.fresh(e5, 125, lambda t: bool(TRIGGER.search(t)))]
    rows += [row(t, 0, "final_holdout", "enron_plain_f", "none") for t in pool.fresh(e5, 125, lambda t: not TRIGGER.search(t))]
    carriers = pool.fresh(e5, 150)
    inj = [(s, "embed_tasktracker_f") for s in pool.fresh(held_out, 100)] + \
          [(s, "embed_llmail_f") for s in pool.fresh(p2, 50, short)]
    for c, (s, g) in zip(carriers, inj):
        text = BD.insert(c, s, rng)
        pool.seen.add(norm(text))
        rows.append(row(text, 1, "final_holdout", g, "embedded", s))
    rows += [row(t, 0, "final_holdout", "enron_clean_f", "none") for t in pool.fresh(e5, 150)]

    path = OUT / "final_holdout.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"rows": len(rows), "injected": sum(r["label"] for r in rows), "sha256": h, "seed": SEED,
                "earlier_texts_checked": n_earlier, "dropped_overlap": pool.dropped,
                "tasktracker_unused_held_out_available": len(held_out)}
    (OUT / "manifest_final.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

"""Build the CONFIRMATORY e-mail set (unseen_email_2), created after the first unseen evaluation.

Purpose: the first unseen e-mail set (unseen_email) produced a finding and a post-hoc hypothesis
(whole-document detector beats the context guard on standalone attack e-mails; a max-combination of the
two recovers recall). Testing a hypothesis on the data that suggested it is circular, so this set is drawn
fresh and is used once, after the hypothesis and every threshold are fixed.

Sources (disjoint from every earlier set, enforced by normalised-text dedup):
  * attacks: LLMail-Inject PHASE 2 labelled submissions (attack_attempt True; api_triggered or judge),
    microsoft/llmail-inject-challenge @1063bdf01ec8762b812d5e06ee768a06faa5a6f7 (MIT). Phase 2 ran against
    newer defences than phase 1, so its attacks are a shifted distribution.
  * benign: Enron e-mails from byte ranges of LLM-PBE/enron-email @a4761f17a981d79bd286bfff7dd0d53645aa5027
    (Apache-2.0) that do not overlap the ranges used for unseen_email.

Output: eval/context_sets/unseen_email_2.jsonl (+ manifest_confirm.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import SOURCES  # noqa: E402  (download_sources.py fills it)

PHASE2 = SOURCES / "llmail2" / "labelled_unique_submissions_phase2.json"
ENRON2 = SOURCES / "enron2"
OUT = Path(os.environ.get("PPAGUARD_CTX_OUT", ROOT / "eval/context_sets"))
SEED = 20261025
N_ATTACK = 400
N_BENIGN = 400


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def existing_norms() -> set[str]:
    s: set[str] = set()
    for p in [ROOT / "eval/sets/validation.jsonl", ROOT / "eval/sets/test.jsonl",
              OUT / "ctx_dev.jsonl", OUT / "unseen_synth.jsonl", OUT / "unseen_email.jsonl"]:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    s.add(norm(json.loads(line)["text"]))
    return s


def main() -> None:
    rng = random.Random(SEED)
    seen = existing_norms()
    n_existing = len(seen)

    data = json.loads(PHASE2.read_text())
    attacks = []
    for key, v in data.items():
        if v.get("attack_attempt") == "True" and v.get("reason") in ("api_triggered", "judge"):
            m = re.search(r"Body:\s*(.*)$", key, flags=re.S)
            b = (m.group(1) if m else key).strip()
            if 20 <= len(b) <= 4000:
                attacks.append(b)
    attacks = list(dict.fromkeys(attacks))
    rng.shuffle(attacks)

    benign = []
    for chunk in sorted(ENRON2.glob("chunk_*.bin")):
        for line in chunk.read_bytes().split(b"\n")[1:-1]:
            try:
                t = json.loads(line).get("text", "").strip()
            except Exception:
                continue
            if 80 <= len(t) <= 3000:
                benign.append(t)
    benign = list(dict.fromkeys(benign))
    rng.shuffle(benign)

    rows, dropped = [], 0

    def take(pool, label, group, n):
        nonlocal dropped
        got = 0
        for t in pool:
            if got >= n:
                break
            k = norm(t)
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            rid = hashlib.sha256((group + "|" + t).encode()).hexdigest()[:16]
            rows.append({"id": rid, "text": t, "label": label, "channel": "data", "carrier": "email",
                         "subset": "unseen_email_2", "attack_style": "llmail_phase2" if label else "none",
                         "group": group, "payload": ""})
            got += 1
        return got

    na = take(attacks, 1, "llmail_phase2", N_ATTACK)
    nb = take(benign, 0, "enron_ham_2", N_BENIGN)
    with open(OUT / "unseen_email_2.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    manifest = {"seed": SEED, "attacks": na, "benign": nb, "dropped_overlap_with_earlier_sets": dropped,
                "earlier_texts_checked": n_existing,
                "sources": {"llmail_phase2": "microsoft/llmail-inject-challenge @1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
                            "enron": "LLM-PBE/enron-email @a4761f17a981d79bd286bfff7dd0d53645aa5027, offsets i*98e6+49e6 (+2e6)"}}
    (OUT / "manifest_confirm.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

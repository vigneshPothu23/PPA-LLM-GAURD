"""Round 2 datasets for the document-aware second stage (written after round 1; see PREREGISTRATION_R2.md).

Development (fitting + thresholds of stage 2 only):
  dev2_email.jsonl
    * 150 LLMail-Inject PHASE-1 attack e-mails (not in unseen_email)                     label 1, llmail_dev
    * 150 Enron e-mails, range enron4                                                    label 0, enron_dev
    * 150 Enron e-mails (enron4, other carriers) with an embedded injection:
        75 held-IN TaskTracker segments not used in ctx_dev + 75 short phase-1 bodies     label 1, embed_dev
    * pool A benign trigger-word sentences as standalone texts                           label 0, trigger_dev

Round-2 test (scored ONCE, after PREREGISTRATION_R2.md is pushed):
  unseen_email_3.jsonl
    * 300 LLMail-Inject PHASE-2 attack e-mails (not in unseen_email_2)                    label 1, llmail_phase2_r3
    * 300 Enron e-mails, range enron3: 150 that naturally contain a trigger word
      (pattern fixed below, before any scoring) and 150 that do not                      label 0, enron_trigger_r3 / enron_plain_r3
  unseen_embed_3.jsonl
    * 150 Enron e-mails (enron3, distinct carriers) with an embedded injection:
        75 held-OUT TaskTracker segments not used in unseen_synth + 75 short phase-2 bodies label 1, embed_tasktracker_r3 / embed_llmail_r3
    * 150 other Enron e-mails (enron3), unmodified                                       label 0, enron_clean_r3

No attack text is authored: injected strings are published TaskTracker segments or LLMail submissions.
Every text is deduplicated (normalised) against the frozen sets, every round-1 set and each other.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_datasets as BD  # noqa: E402
import pools_a as A  # noqa: E402
from paths import SOURCES  # noqa: E402

OUT = Path(os.environ.get("PPAGUARD_CTX_OUT", ROOT / "eval/context_sets"))
SEED = 20261026
TRIGGER = re.compile(r"\b(ignore|disregard|override|password|delete|system|forward|execute|admin|bypass|"
                     r"instructions?)\b", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def enron(dirname: str) -> list[str]:
    out = []
    for chunk in sorted((SOURCES / dirname).glob("chunk_*.bin")):
        for line in chunk.read_bytes().split(b"\n")[1:-1]:
            try:
                t = json.loads(line).get("text", "").strip()
            except Exception:
                continue
            if 80 <= len(t) <= 3000:
                out.append(t)
    return list(dict.fromkeys(out))


def llmail_bodies(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    out = []
    for key, v in data.items():
        if v.get("attack_attempt") == "True" and v.get("reason") in ("api_triggered", "judge"):
            m = re.search(r"Body:\s*(.*)$", key, flags=re.S)
            b = (m.group(1) if m else key).strip()
            if 20 <= len(b) <= 4000:
                out.append(b)
    return list(dict.fromkeys(out))


class Pool:
    def __init__(self, seen: set[str]):
        self.seen = seen
        self.dropped = 0

    def fresh(self, items, n, extra=lambda t: True):
        got = []
        for t in items:
            if len(got) >= n:
                break
            k = norm(t)
            if k in self.seen or not extra(t):
                if k in self.seen:
                    self.dropped += 1
                continue
            self.seen.add(k)
            got.append(t)
        if len(got) < n:
            raise SystemExit(f"not enough fresh items ({len(got)}/{n})")
        return got


def row(text, label, subset, group, style, payload=""):
    rid = hashlib.sha256((subset + "|" + group + "|" + text).encode()).hexdigest()[:16]
    return {"id": rid, "text": text, "label": label, "channel": "data", "carrier": "email", "subset": subset,
            "attack_style": style, "group": group, "payload": payload}


def main() -> None:
    rng = random.Random(SEED)
    earlier = [ROOT / "eval/sets/validation.jsonl", ROOT / "eval/sets/test.jsonl"] + \
              [OUT / f"{n}.jsonl" for n in ("ctx_dev", "unseen_synth", "unseen_email", "unseen_email_2")]
    seen = {norm(r["text"]) for p in earlier for r in read_jsonl(p)}
    n_earlier = len(seen)
    pool = Pool(seen)

    # TaskTracker segments: same hash split as round 1, minus the strings round 1 actually used
    all_inj = BD.recover_tasktracker_injections()
    odd = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2  # noqa: E731
    used = {norm(r["payload"]) for n in ("ctx_dev", "unseen_synth") for r in read_jsonl(OUT / f"{n}.jsonl") if r.get("payload")}
    held_in = [s for s in all_inj if odd(s) == 0 and norm(s) not in used]
    held_out = [s for s in all_inj if odd(s) == 1 and norm(s) not in used]
    rng.shuffle(held_in)
    rng.shuffle(held_out)

    p1 = llmail_bodies(SOURCES / "llmail" / "labelled_unique_submissions_phase1.json")
    p2 = llmail_bodies(SOURCES / "llmail2" / "labelled_unique_submissions_phase2.json")
    rng.shuffle(p1)
    rng.shuffle(p2)
    e3, e4 = enron("enron3"), enron("enron4")
    rng.shuffle(e3)
    rng.shuffle(e4)
    short = lambda t: 40 <= len(t) <= 400  # noqa: E731

    # ---------------- development ----------------
    dev = []
    dev += [row(t, 1, "dev2_email", "llmail_dev", "llmail_phase1") for t in pool.fresh(p1, 150)]
    dev += [row(t, 0, "dev2_email", "enron_dev", "none") for t in pool.fresh(e4, 150)]
    carriers = pool.fresh(e4, 150)
    inj = held_in[:75] + pool.fresh(p1, 75, short)
    for c, s in zip(carriers, inj):
        text = BD.insert(c, s, rng)
        pool.seen.add(norm(text))
        dev.append(row(text, 1, "dev2_email", "embed_dev", "embedded", s))
    trig = [t for items in A.BENIGN_TRIGGER.values() for t in items]
    dev += [row(t, 0, "dev2_email", "trigger_dev", "benign_trigger") for t in pool.fresh(trig, len(trig))]

    # ---------------- round-2 test ----------------
    t3 = [row(t, 1, "unseen_email_3", "llmail_phase2_r3", "llmail_phase2") for t in pool.fresh(p2, 300)]
    t3 += [row(t, 0, "unseen_email_3", "enron_trigger_r3", "none") for t in pool.fresh(e3, 150, lambda t: bool(TRIGGER.search(t)))]
    t3 += [row(t, 0, "unseen_email_3", "enron_plain_r3", "none") for t in pool.fresh(e3, 150, lambda t: not TRIGGER.search(t))]

    b3 = []
    carriers = pool.fresh(e3, 150)
    inj = [(s, "embed_tasktracker_r3") for s in held_out[:75]] + [(s, "embed_llmail_r3") for s in pool.fresh(p2, 75, short)]
    for c, (s, g) in zip(carriers, inj):
        text = BD.insert(c, s, rng)
        pool.seen.add(norm(text))
        b3.append(row(text, 1, "unseen_embed_3", g, "embedded", s))
    b3 += [row(t, 0, "unseen_embed_3", "enron_clean_r3", "none") for t in pool.fresh(e3, 150)]

    out = {}
    for name, rows in (("dev2_email", dev), ("unseen_email_3", t3), ("unseen_embed_3", b3)):
        with open(OUT / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        h = hashlib.sha256((OUT / f"{name}.jsonl").read_bytes()).hexdigest()
        out[name] = {"rows": len(rows), "injected": sum(r["label"] for r in rows), "sha256": h}
        print(f"{name}: {len(rows)} rows ({out[name]['injected']} injected) sha256 {h[:16]}")
    out.update({"seed": SEED, "earlier_texts_checked": n_earlier, "dropped_overlap": pool.dropped,
                "tasktracker_unused_held_in": len(held_in), "tasktracker_unused_held_out": len(held_out),
                "trigger_pattern": TRIGGER.pattern})
    (OUT / "manifest_round2.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

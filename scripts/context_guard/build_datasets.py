"""Build the context-guard evaluation datasets.

Outputs (written to the directory given by --out, default eval/context_sets/, which is git-ignored like
eval/sets/):

  ctx_dev.jsonl        development / tuning set (carriers from pool A + held-in TaskTracker injections +
                       pool A benign trigger-word content). Used ONLY for fitting and threshold selection.
  unseen_synth.jsonl   unseen synthetic set (carriers from pool B + held-OUT TaskTracker injections +
                       pool B benign trigger-word content). Never tuned on.
  unseen_email.jsonl   unseen real email set (LLMail-Inject attack emails + LLMail FP emails + Enron ham).
                       Never tuned on.

Each row: {id, text, label, channel, carrier, subset, attack_style, group}.
  label 1 = contains an injected / out-of-task instruction aimed at an AI reader.
  label 0 = benign (clean carrier, or benign trigger-word content, or a real benign email).
  channel = "data" for everything here (all of it is untrusted document content).

Discipline:
  * Injected PAYLOAD text is never authored in this repo. It comes from published datasets:
      - TaskTracker injected segments recovered from the audited training candidate (clean/poisoned diff).
      - LLMail-Inject attack email bodies (microsoft/llmail-inject-challenge, MIT).
  * The TaskTracker injection pool is split by hash into held-in (ctx_dev) and held-out (unseen_synth);
    the two never share an injection string.
  * A benign carrier is never reused as both a clean example and the base of an injected example within
    the same subset row id space; carriers are drawn round-robin.
  * Every produced text is checked for normalised-text overlap against the frozen dev and test sets, and
    against the other produced splits. Any collision is dropped and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import pools_a as A  # noqa: E402
import pools_b as B  # noqa: E402

FROZEN_DEV = REPO / "eval" / "sets" / "validation.jsonl"
FROZEN_TEST = REPO / "eval" / "sets" / "test.jsonl"
TRAIN_CANDIDATE = REPO / "experiments" / "results" / "injecguard_audit_v2" / "candidate_train.jsonl"
from paths import SOURCES  # noqa: E402  (download_sources.py fills it)

LLMAIL = SOURCES / "llmail" / "labelled_unique_submissions_phase1.json"
LLMAIL_FP = SOURCES / "llmail" / "emails_for_fp_tests.json"
ENRON_CHUNKS = SOURCES / "enron"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_frozen_norms() -> set[str]:
    norms: set[str] = set()
    for path in (FROZEN_DEV, FROZEN_TEST):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    norms.add(norm(json.loads(line)["text"]))
    return norms


def recover_tasktracker_injections() -> list[str]:
    """Recover injected segments as the diff between poisoned and matched clean TaskTracker texts."""
    with open(TRAIN_CANDIDATE) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    tt = [r for r in rows if r.get("source") == "TaskTracker"]
    clean = [r["text"].strip() for r in tt if r["label"] == 0]
    pois = [r["text"].strip() for r in tt if r["label"] == 1]
    byhead: dict[str, list[str]] = {}
    bytail: dict[str, list[str]] = {}
    for t in clean:
        byhead.setdefault(t[:80], []).append(t)
        bytail.setdefault(t[-80:], []).append(t)
    seen: set[str] = set()
    injections: list[str] = []
    for p in pois:
        base = None
        for c in byhead.get(p[:80], []) + bytail.get(p[-80:], []):
            if c in p and len(p) > len(c) + 10:
                base = c
                break
        if not base:
            continue
        i = p.find(base)
        seg = (p[:i] + " " + p[i + len(base):]).strip()
        seg = re.sub(r"\s+", " ", seg)
        if 12 <= len(seg) <= 600 and norm(seg) not in seen:
            seen.add(norm(seg))
            injections.append(seg)
    return injections


def load_llmail(n_attack: int, rng: random.Random) -> tuple[list[str], list[str]]:
    """Return (attack_bodies, fp_emails). Attacks: api_triggered or judge==True. FP: emails_for_fp_tests."""
    data = json.loads(LLMAIL.read_text())

    def body_of(key: str) -> str:
        # keys look like: "Subject of the email: X.   Body: Y"
        m = re.search(r"Body:\s*(.*)$", key, flags=re.S)
        return (m.group(1) if m else key).strip()

    attacks = []
    for key, v in data.items():
        aa = v.get("attack_attempt")
        reason = v.get("reason")
        if aa == "True" and reason in ("api_triggered", "judge"):
            b = body_of(key)
            if 20 <= len(b) <= 4000:
                attacks.append(b)
    attacks = list(dict.fromkeys(attacks))
    rng.shuffle(attacks)
    attacks = attacks[:n_attack]

    fp_raw = json.loads(LLMAIL_FP.read_text())
    fps = []
    for e in fp_raw:
        s = e if isinstance(e, str) else json.dumps(e)
        fps.append(s.strip())
    return attacks, fps


def load_enron_ham(n: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for chunk in sorted(ENRON_CHUNKS.glob("chunk_*.bin")):
        raw = chunk.read_bytes().split(b"\n")[1:-1]
        for line in raw:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get("text", "").strip()
            if 80 <= len(t) <= 3000:
                out.append(t)
    out = list(dict.fromkeys(out))
    rng.shuffle(out)
    return out[:n]


def render_table(spec: dict, rng: random.Random, inject: str | None, n_rows: int = 5) -> str:
    cols = spec["columns"]
    vals = spec["values"]
    lines = [" | ".join(cols), " | ".join("---" for _ in cols)]
    rows = []
    for _ in range(n_rows):
        rows.append([rng.choice(vals[c]) for c in cols])
    if inject is not None:
        # place the injected instruction in a cell of a random row (last column)
        r = rng.randrange(len(rows))
        rows[r][-1] = inject
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def insert(carrier: str, inject: str, rng: random.Random) -> str:
    where = rng.choice(["end", "start", "middle"])
    if where == "end":
        return carrier.rstrip() + "\n\n" + inject
    if where == "start":
        return inject + "\n\n" + carrier.lstrip()
    paras = carrier.split("\n\n")
    if len(paras) < 2:
        return carrier.rstrip() + "\n\n" + inject
    k = rng.randrange(1, len(paras))
    return "\n\n".join(paras[:k] + [inject] + paras[k:])


class Builder:
    def __init__(self, frozen: set[str], seed: int):
        self.frozen = frozen
        self.produced: set[str] = set()
        self.rng = random.Random(seed)
        self.dropped = 0

    def add(self, rows: list[dict], text: str, label: int, channel: str, carrier: str,
            subset: str, style: str, group: str, payload: str = ""):
        n = norm(text)
        if n in self.frozen or n in self.produced:
            self.dropped += 1
            return
        self.produced.add(n)
        rid = hashlib.sha256((subset + "|" + text).encode()).hexdigest()[:16]
        rows.append({"id": rid, "text": text, "label": label, "channel": channel,
                     "carrier": carrier, "subset": subset, "attack_style": style, "group": group,
                     "payload": payload})


def build_synth(builder: Builder, carriers, benign_trigger, task_irrelevant, injections,
                subset: str) -> list[dict]:
    rng = builder.rng
    rows: list[dict] = []
    emails, documents, tables, code = carriers
    inj = list(injections)
    rng.shuffle(inj)
    ii = 0

    def next_inj() -> str:
        nonlocal ii
        s = inj[ii % len(inj)]
        ii += 1
        return s

    # Clean carriers (label 0)
    for c in emails:
        builder.add(rows, c, 0, "data", "email", subset, "none", "clean")
    for c in documents:
        builder.add(rows, c, 0, "data", "document", subset, "none", "clean")
    for c in code:
        builder.add(rows, c, 0, "data", "code", subset, "none", "clean")
    for spec in tables:
        builder.add(rows, render_table(spec, rng, None), 0, "data", "table", subset, "none", "clean")

    # Benign trigger-word content (label 0)
    for carrier_name, items in benign_trigger.items():
        for t in items:
            if carrier_name == "table":
                base = render_table(rng.choice(tables), rng, None)
                text = base + "\n" + " | ".join([t] + ["" for _ in range(2)])
            elif carrier_name == "code":
                text = rng.choice(code).rstrip() + "\n" + t + "\n"
            elif carrier_name == "email":
                text = insert(rng.choice(emails), t, rng)
            else:
                text = insert(rng.choice(documents), t, rng)
            builder.add(rows, text, 0, "data", carrier_name, subset, "benign_trigger", "trigger")

    # Injected carriers (label 1): published injection strings embedded in benign carriers.
    for c in emails:
        s = next_inj()
        builder.add(rows, insert(c, s, rng), 1, "data", "email", subset, "published_injection", "email", s)
    for c in documents:
        s = next_inj()
        builder.add(rows, insert(c, s, rng), 1, "data", "document", subset, "published_injection", "document", s)
    for spec in tables:
        s = next_inj()
        builder.add(rows, render_table(spec, rng, s), 1, "data", "table", subset, "published_injection", "table", s)
    for c in code:
        seg = next_inj()
        text = c.rstrip() + "\n# " + seg + "\n"
        builder.add(rows, text, 1, "data", "code", subset, "published_injection", "code", seg)

    # Task-irrelevant (harmless but off-task) instructions appended to documents (label 1).
    for t in task_irrelevant:
        builder.add(rows, insert(rng.choice(documents), t, rng), 1, "data", "document", subset,
                    "task_irrelevant", "task_irrelevant", t)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "eval" / "context_sets"))
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--n-llmail-attack", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_norms()

    all_inj = recover_tasktracker_injections()
    # split by hash: held-in (ctx_dev) vs held-out (unseen_synth)
    held_in = [s for s in all_inj if int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2 == 0]
    held_out = [s for s in all_inj if int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2 == 1]

    builder = Builder(frozen, args.seed)

    ctx_dev = build_synth(
        builder,
        ([*A.EMAILS], [*A.DOCUMENTS], [*A.TABLES], [*A.CODE]),
        A.BENIGN_TRIGGER, A.TASK_IRRELEVANT, held_in, "ctx_dev",
    )
    unseen_synth = build_synth(
        builder,
        ([*B.EMAILS], [*B.DOCUMENTS], [*B.TABLES], [*B.CODE]),
        B.BENIGN_TRIGGER, [], held_out, "unseen_synth",
    )

    # Unseen real emails
    rng = builder.rng
    attacks, fps = load_llmail(args.n_llmail_attack, rng)
    ham = load_enron_ham(len(fps) + 200, rng)
    unseen_email: list[dict] = []
    for b in attacks:
        builder.add(unseen_email, b, 1, "data", "email", "unseen_email", "llmail_attack", "llmail")
    for b in fps:
        builder.add(unseen_email, b, 0, "data", "email", "unseen_email", "none", "llmail_fp")
    for b in ham:
        builder.add(unseen_email, b, 0, "data", "email", "unseen_email", "none", "enron_ham")

    def dump(name: str, rows: list[dict]) -> None:
        with open(out / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in rows)
        print(f"{name}: {len(rows)} rows ({pos} injected / {len(rows) - pos} benign)")

    dump("ctx_dev.jsonl", ctx_dev)
    dump("unseen_synth.jsonl", unseen_synth)
    dump("unseen_email.jsonl", unseen_email)

    manifest = {
        "seed": args.seed,
        "tasktracker_injections_total": len(all_inj),
        "held_in": len(held_in),
        "held_out": len(held_out),
        "llmail_attack_sampled": len(attacks),
        "llmail_fp": len(fps),
        "enron_ham": len(ham),
        "dropped_collisions": builder.dropped,
        "frozen_norms": len(frozen),
        "sources": {
            "tasktracker": "experiments/results/injecguard_audit_v2/candidate_train.jsonl (audited v2 candidate)",
            "llmail": "microsoft/llmail-inject-challenge @1063bdf01ec8762b812d5e06ee768a06faa5a6f7 (MIT)",
            "enron": "LLM-PBE/enron-email @a4761f17a981d79bd286bfff7dd0d53645aa5027 (Apache-2.0)",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps(manifest))


if __name__ == "__main__":
    main()

"""Build eval/scenarios/scenarios.jsonl from scenario_data.py (deterministic; no randomness, no downloads).

    python eval/scenarios/build_scenarios.py

Row: {id, split, label, category, channel, task, carrier, payload, payload_text, text, expected, provenance}
Also writes scenarios_dev.jsonl and scenarios_holdout.jsonl (the same rows, one file per split).
  label 1 = contains an injected instruction aimed at the AI; expected = "flag" (WARN or BLOCK, never forwarded)
  label 0 = benign;                                             expected = "allow" (forwarded with Engine 1 wrapping)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from scenario_data import BENIGN_DIRECT, BENIGN_TRIGGER_LINES, CARRIERS, PAYLOADS  # noqa: E402

OUT = HERE / "scenarios.jsonl"
PROVENANCE = "synthetic: hand-written for PPA Guard (eval/scenarios/scenario_data.py); not real-world data"
BENIGN_CAT = {"readme": "repo_summary", "email": "technical_email", "code": "code_review", "cicd_log": "ci_troubleshooting",
              "table": "documentation_table", "config": "config_troubleshooting"}
CHANNEL_CAT = {"readme_embedded": "readme", "email_embedded": "email", "code_review": "code", "cicd_log": "cicd_log",
               "table_markdown": "table"}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def fill(carrier: dict, line: str) -> str:
    """Insert `line` at the carrier's slot, adapting it to the channel (comment syntax, indentation, table cells)."""
    text = carrier["text"]
    ch = carrier["channel"]
    if ch == "table":
        line = " ".join(part.strip() for part in line.splitlines())
    elif carrier.get("comment") and not line.lstrip().startswith(carrier["comment"]):
        line = "\n".join(f"{carrier['comment']} {part}" for part in line.splitlines())
    before = text.split("{SLOT}")[0]
    indent = re.match(r"[ \t]*", before.rsplit("\n", 1)[-1]).group(0)
    line = line.replace("\n", "\n" + indent)
    return text.replace("{SLOT}", line)


def main() -> None:
    rows: list[dict] = []

    def add(split, label, category, channel, task, carrier, payload, text, payload_text=""):
        rid = hashlib.sha256(f"{split}|{category}|{carrier}|{payload}|{text}".encode()).hexdigest()[:12]
        rows.append(dict(id=rid, split=split, label=label, category=category, channel=channel, task=task, carrier=carrier,
                         payload=payload, payload_text=payload_text, text=text, expected="flag" if label else "allow",
                         provenance=PROVENANCE))

    by_split = {s: [c for c in CARRIERS if c["split"] == s] for s in ("dev", "holdout")}
    # benign: every carrier with its own benign line
    for c in CARRIERS:
        add(c["split"], 0, BENIGN_CAT[c["channel"]], c["channel"], c["task"], c["id"], "", fill(c, c["benign"]))
    # benign: trigger-word lines inside carriers of the same channel and split (round-robin)
    for ch, lines in BENIGN_TRIGGER_LINES.items():
        for k, (split, line) in enumerate(lines):
            pool = [c for c in by_split[split] if c["channel"] == ch]
            c = pool[k % len(pool)]
            add(split, 0, "trigger_word_benign", ch, c["task"], c["id"], "", fill(c, line))
    # benign: ordinary developer requests (no carrier)
    for b in BENIGN_DIRECT:
        add(b["split"], 0, b["cat"], "direct", "Answer the user's question.", "", "", b["text"])

    # attacks
    pid = Counter()
    for p in PAYLOADS:
        cat, split = p["cat"], p["split"]
        pid[cat] += 1
        payload_id = f"{cat}-{split}-{pid[cat]}"
        if cat == "direct_override":
            add(split, 1, cat, "direct", "Answer the user's question.", "", payload_id, p["text"], p["text"])
            continue
        if cat in CHANNEL_CAT:
            targets = [c for c in by_split[split] if c["channel"] == CHANNEL_CAT[cat]]
        else:  # generic categories: three carriers from different channels, rotated per payload
            pool = sorted(by_split[split], key=lambda c: (c["channel"], c["id"]))
            start = (pid[cat] * 5) % len(pool)
            targets, seen = [], set()
            for c in pool[start:] + pool[:start]:
                if c["channel"] not in seen:
                    targets.append(c)
                    seen.add(c["channel"])
                if len(targets) == 3:
                    break
        for c in targets:
            add(split, 1, cat, c["channel"], c["task"], c["id"], payload_id, fill(c, p["text"]), p["text"])

    # integrity: unique texts, split-disjoint carriers/payloads, no overlap with the frozen sets
    texts = [norm(r["text"]) for r in rows]
    assert len(texts) == len(set(texts)), "duplicate scenario text"
    frozen = set()
    for f in ("validation.jsonl", "test.jsonl"):
        p = ROOT / "eval" / "sets" / f
        if p.exists():
            frozen |= {norm(json.loads(l)["text"]) for l in open(p, encoding="utf-8") if l.strip()}
    overlap = sum(t in frozen for t in texts)
    assert overlap == 0, f"{overlap} scenario texts overlap the frozen sets"
    for key in ("carrier", "payload"):
        dev = {r[key] for r in rows if r["split"] == "dev" and r[key]}
        hold = {r[key] for r in rows if r["split"] == "holdout" and r[key]}
        assert not dev & hold, f"{key} shared across splits"

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("dev", "holdout"):
        with open(HERE / f"scenarios_{split}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                if r["split"] == split:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["split"], r["label"]) for r in rows)
    print(f"{OUT.name}: {len(rows)} rows  sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    print(f"  dev: {c[('dev', 1)]} attack / {c[('dev', 0)]} benign   holdout: {c[('holdout', 1)]} attack / {c[('holdout', 0)]} benign")
    print("  frozen-set overlap checked:", bool(frozen))
    for split in ("dev", "holdout"):
        print(f"  {split}:", dict(sorted(Counter(r['category'] for r in rows if r['split'] == split).items())))


if __name__ == "__main__":
    main()

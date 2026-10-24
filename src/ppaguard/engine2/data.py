"""Data ingestion and validation for Engine 2.

Accepted files: JSONL, JSON (list of objects) or CSV with at least `text` and `label` columns. Labels may be 0/1,
"benign"/"injection" (also "safe", "jailbreak", "prompt_injection", ...). Optional: `id`, `source`, `group`.
Nothing here edits the input files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("benign", "injection")
_LABEL_ALIASES = {
    "0": 0, "benign": 0, "safe": 0, "legit": 0, "legitimate": 0, "clean": 0, "negative": 0, "false": 0,
    "1": 1, "injection": 1, "prompt_injection": 1, "jailbreak": 1, "malicious": 1, "attack": 1, "unsafe": 1,
    "positive": 1, "true": 1,
}


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    label: int
    source: str = ""
    group: str = ""


@dataclass
class ValidationReport:
    path: str
    file_sha256: str
    rows_read: int = 0
    rows_valid: int = 0
    issues: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)

    def note(self, kind: str, where: str) -> None:
        self.issues[kind] += 1
        self.examples.setdefault(kind, where)

    def as_dict(self) -> dict:
        return {"path": self.path, "file_sha256": self.file_sha256, "rows_read": self.rows_read,
                "rows_valid": self.rows_valid, "issues": dict(self.issues), "examples": self.examples}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", t).strip()


def text_key(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8", "surrogatepass")).hexdigest()


def parse_label(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        return _LABEL_ALIASES.get(value.strip().lower())
    return None


def _rows(path: Path):
    suf = path.suffix.lower()
    if suf == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield i, json.loads(line)
                    except json.JSONDecodeError:
                        yield i, None
    elif suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("rows") or data.get("records")
        if not isinstance(data, list):
            raise DataError(f"{path}: expected a JSON list of records")
        yield from enumerate(data, 1)
    elif suf == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            yield from enumerate(csv.DictReader(f), 1)
    else:
        raise DataError(f"{path}: unsupported file type {suf}")


def load_records(path: str | Path, text_field: str = "text", label_field: str = "label",
                 group_flags: str | Path | None = None) -> tuple[list[Record], ValidationReport]:
    """Read and validate. Invalid rows are skipped and counted, never repaired silently."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{path}: file not found")
    rep = ValidationReport(str(path), sha256_file(path))
    groups = _load_group_flags(group_flags) if group_flags else {}
    out: list[Record] = []
    for i, row in _rows(path):
        rep.rows_read += 1
        where = f"row {i}"
        if not isinstance(row, dict):
            rep.note("not_an_object", where)
            continue
        text = row.get(text_field)
        if not isinstance(text, str):
            rep.note("text_missing_or_not_string", where)
            continue
        if not text.strip():
            rep.note("text_empty", where)
            continue
        if "\x00" in text:
            rep.note("text_contains_nul", where)
            continue
        label = parse_label(row.get(label_field))
        if label is None:
            rep.note("label_invalid", where)
            continue
        rid = str(row.get("id") or f"{path.stem}:{i}")
        group = str(row.get("group") or groups.get(rid) or "")
        out.append(Record(rid, text, label, str(row.get("source") or row.get("set") or ""), group))
    rep.rows_valid = len(out)
    if not out:
        raise DataError(f"{path}: no valid records")
    return out, rep


def _load_group_flags(path) -> dict[str, str]:
    """The v2 audit marks paired clean/poisoned variants with a `pair:<id>` flag; such rows must share a split."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                for fl in r.get("flags", []):
                    if str(fl).startswith("pair:"):
                        out[r["id"]] = fl
    return out


def dedup_and_conflicts(records: list[Record]) -> tuple[list[Record], dict]:
    """Drops exact (normalised) duplicates, keeping the first, and every row of a text that carries both labels."""
    by_key: dict[str, set[int]] = defaultdict(set)
    for r in records:
        by_key[text_key(r.text)].add(r.label)
    conflicted = {k for k, labels in by_key.items() if len(labels) > 1}
    seen, kept, dup, conf = set(), [], 0, 0
    for r in records:
        k = text_key(r.text)
        if k in conflicted:
            conf += 1
            continue
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        kept.append(r)
    return kept, {"duplicates_removed": dup, "conflicting_rows_removed": conf, "conflicting_texts": len(conflicted)}


def assert_no_leakage(train: list[Record], references: dict[str, list[Record]]) -> dict:
    """Refuses training data that shares a normalised text with any frozen evaluation set."""
    tk = {text_key(r.text) for r in train}
    overlaps = {name: sum(1 for r in recs if text_key(r.text) in tk) for name, recs in references.items()}
    if any(overlaps.values()):
        raise DataError(f"training data overlaps frozen evaluation data: {overlaps}")
    return overlaps


def grouped_split(records: list[Record], val_fraction: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Group-aware, label-stratified split: all rows of one group land on the same side."""
    if not 0 < val_fraction < 0.5:
        raise DataError("val_fraction must be in (0, 0.5)")
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.group or ("t:" + text_key(r.text))].append(r)
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for g, rows in groups.items():
        by_label[max(x.label for x in rows)].append(g)
    val_groups = set()
    for label, gs in sorted(by_label.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        target = round(sum(len(groups[g]) for g in gs) * val_fraction)
        n = 0
        for g in gs:
            if n >= target:
                break
            val_groups.add(g)
            n += len(groups[g])
    train = [r for g, rows in groups.items() if g not in val_groups for r in rows]
    val = [r for g, rows in groups.items() if g in val_groups for r in rows]
    return train, val


def class_balance(records: list[Record]) -> dict:
    c = Counter(r.label for r in records)
    n = len(records)
    return {"rows": n, **{LABELS[k]: c.get(k, 0) for k in (0, 1)},
            "injection_share": round(c.get(1, 0) / n, 4) if n else None}


def manifest(records: list[Record]) -> str:
    """Order-independent fingerprint of (text, label) content."""
    h = hashlib.sha256()
    for k in sorted(f"{text_key(r.text)}:{r.label}" for r in records):
        h.update(k.encode())
    return h.hexdigest()

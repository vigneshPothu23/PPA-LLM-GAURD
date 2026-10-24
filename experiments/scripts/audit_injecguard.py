"""Audit InjecGuard/PIGuard training data and, only if the gate allows, prepare a CANDIDATE fine-tuning set.

This script never trains, never edits its inputs and never decides whether fine-tuning happens.
It produces evidence for that decision.

Inputs (all opened read-only; their SHA-256 is recorded before and re-checked after the run):
  --candidates     JSON list of records (InjecGuard datasets/train.json)
  --label-evidence InjecGuard repository root; eval.py / train.py / util.py are read to confirm the label mapping
  --dev            frozen development set (eval/sets/validation.jsonl) -- the threshold-selection split
  --frozen-test    frozen test set (eval/sets/test.jsonl) -- read only to detect overlap, never used otherwise
  --extra-dev      optional further dev sets that must stay disjoint (e.g. experiments/data/dev_indirect_v2.jsonl)
  --bipia          optional BIPIA repo root; its TEST contexts and TEST attack pools are checked for containment
  --eval-loader    optional eval/compare.py; confirms which field of the dev/test rows is passed to Engine 2

Outputs (written only into --out, which must be new or a directory this script created earlier):
  audit_report.json, AUDIT_SUMMARY.md, exclusion_manifest.jsonl, dataset_manifest.json,
  candidate_train.jsonl (only when the gate is PASS or PASS WITH WARNINGS)

Gate:
  FAIL               critical problem; no candidate is written; exit status 2
  PASS WITH WARNINGS no blocking problem, documented issues need human review; exit status 0
  PASS               no issues; exit status 0
  A PASS of either kind is NOT an approval to fine-tune.

Normalisation (used only to compare records; output text is always the original string):
  key   = NFKC -> casefold -> collapse all whitespace to one space -> strip
  loose = key with every character that is not a letter or digit removed (near-duplicate key)
  shingles = word 3-grams of the normalised text; Jaccard / containment against dev and test texts
              (shingles found in more than POSTING_CAP references are boilerplate and ignored on both sides)
  verbatim  = every reference of >= VERBATIM_MIN normalised characters is also searched for inside each candidate
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
MARKER = ".audit_outputs.json"
OUTPUT_FILES = ("audit_report.json", "AUDIT_SUMMARY.md", "exclusion_manifest.jsonl", "dataset_manifest.json",
                "candidate_train.jsonl")

# Label mapping for InjecGuard data, with the evidence that is re-verified at run time (see verify_label_evidence).
LABEL_MAPPING = {0: "benign", 1: "injection"}
LABEL_EVIDENCE = [
    ("eval.py", r'if target_class == "benign":\s*\n\s*target_label = 0\s*\n\s*else:\s*\n\s*target_label = 1',
     "acc_compute() scores benign sets against label 0 and injection sets against label 1"),
    ("util.py", r"label_0_indices = labels == 0[\s\S]*accuracy_0 = correct_0 / total_0[\s\S]*return accuracy_0, accuracy_1, total_accuracy",
     "compute_accuracy() returns (accuracy on label 0, accuracy on label 1, total)"),
    ("train.py", r"benign_accuracy, injection_accuracy, total_accuracy = compute_accuracy\(",
     "train.py unpacks that tuple as (benign_accuracy, injection_accuracy, ...): label 0 = benign, 1 = injection"),
    ("eval.py", r'def wildguard_eval[\s\S]*?target_class="benign"', "wildguard.json (all label 0) is evaluated as benign"),
]
# sources whose construction method is close to a frozen evaluation set (reported for human review, not excluded)
REVIEW_SOURCES = {
    "over-defense": "benign prompts with trigger words; same idea as NotInject (frozen validation/test)",
    "BIPIA": "built from BIPIA contexts and attacks; the frozen test uses BIPIA test contexts and attack pools",
}
LONG_CHARS = 2000          # ~512 DeBERTa tokens for English prose; a proxy, the real tokenizer is not used here
NEAR_JACCARD = 0.80        # near-duplicate if word-3-gram Jaccard >= this
CONTAIN = 0.90             # containment if >= this share of the smaller side's shingles appear in the other
MIN_SHINGLES = 5           # below this, only exact / loose keys are used (shingles are too few to be reliable)
POSTING_CAP = 200          # ignore shingles shared by more reference texts than this (boilerplate)
VERBATIM_MIN = 40          # references at least this long (normalised chars) are also searched for verbatim
TARGET_LEAK_PATTERNS = {
    "label_assignment": r"\blabel\s*[:=]\s*[01]\b",
    "json_label_key": r'"label"\s*:',
    "class_word_assignment": r"\b(class|category)\s*[:=]\s*(benign|malicious|injection|safe)\b",
}
TEMPLATE_PATTERN = re.compile(r"^\s*(###\s*\w+|<\|im_start\|>|\[INST\]|Human:|User:|Instruction:|System:)", re.I)


# ----------------------------------------------------------------------------- helpers
def key(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def loose(text: str) -> str:
    return "".join(ch for ch in key(text) if ch.isalnum())


def shingles(text: str) -> frozenset:
    toks = re.findall(r"\w+", key(text))
    return frozenset(" ".join(toks[i:i + 3]) for i in range(len(toks) - 2))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CriticalError(Exception):
    pass


@dataclass
class Gate:
    critical: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def fail(self, code, detail):
        self.critical.append({"code": code, "detail": detail})

    def warn(self, code, detail):
        self.warnings.append({"code": code, "detail": detail})

    @property
    def status(self):
        return "FAIL" if self.critical else ("PASS WITH WARNINGS" if self.warnings else "PASS")


# ----------------------------------------------------------------------------- loading
def load_records(path: Path) -> list:
    """JSON list or JSONL; returns the raw objects. Raises CriticalError on unreadable files."""
    try:
        raw = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise CriticalError(f"{path}: cannot read as UTF-8 ({type(e).__name__})")
    if path.suffix == ".jsonl":
        out = []
        for n, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    raise CriticalError(f"{path}: line {n} is not valid JSON")
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise CriticalError(f"{path}: not valid JSON")
    if not isinstance(data, list):
        raise CriticalError(f"{path}: expected a JSON list of records, found {type(data).__name__}")
    return data


def profile_schema(records: list) -> dict:
    keysets = collections.Counter(tuple(sorted(r)) if isinstance(r, dict) else ("<not an object>",) for r in records)
    fields = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            fields.setdefault(k, collections.Counter())[type(v).__name__] += 1
    return {"records": len(records), "key_sets": [[list(k), n] for k, n in keysets.most_common()],
            "field_types": {k: dict(v) for k, v in sorted(fields.items())}}


def infer_fields(records: list, share: float = 0.5) -> tuple:
    """Text field = field holding a string in at least `share` of records, with the largest mean length;
    label field = field holding a non-bool int in >= `share` of records, with <= 10 distinct values.
    The tolerance lets a few broken records be reported individually instead of hiding the schema."""
    objs = [r for r in records if isinstance(r, dict)]
    if not objs:
        return None, None
    names = sorted({k for r in objs for k in r})
    text, label, best = None, None, -1.0
    for k in names:
        strs = [r[k] for r in objs if isinstance(r.get(k), str)]
        ints = [r[k] for r in objs if isinstance(r.get(k), int) and not isinstance(r.get(k), bool)]
        if len(strs) >= share * len(objs):
            m = sum(len(v) for v in strs) / len(strs)
            if m > best:
                text, best = k, m
        elif len(ints) >= share * len(objs) and len(set(ints)) <= 10:
            label = k
    return text, label


def verify_label_evidence(repo: Path | None, gate: Gate) -> list:
    out = []
    if repo is None:
        gate.fail("LABEL_EVIDENCE_MISSING", "no --label-evidence repository given; the label mapping cannot be verified")
        return out
    for fname, pattern, meaning in LABEL_EVIDENCE:
        p = repo / fname
        ok = p.is_file() and re.search(pattern, p.read_text(encoding="utf-8", errors="replace")) is not None
        out.append({"file": fname, "found": ok, "meaning": meaning, "file_sha256": file_sha(p) if p.is_file() else None})
        if not ok:
            gate.fail("LABEL_EVIDENCE_NOT_FOUND", f"{fname}: expected code establishing the label mapping was not found")
    return out


def verify_eval_loader(path: Path | None, text_field: str, label_field: str, gate: Gate) -> dict:
    if path is None:
        gate.warn("EVAL_LOADER_NOT_CHECKED", "no --eval-loader given; the dev text field was not cross-checked against the evaluation code")
        return {"checked": False}
    src = path.read_text(encoding="utf-8", errors="replace")
    t = re.search(rf'r\["{re.escape(text_field)}"\]', src) is not None
    lb = re.search(rf'r\["{re.escape(label_field)}"\]', src) is not None
    if not (t and lb):
        gate.fail("DEV_FIELD_UNCONFIRMED", f"{path.name} does not read r['{text_field}'] and r['{label_field}']")
    return {"checked": True, "file": str(path), "reads_text_field": t, "reads_label_field": lb, "file_sha256": file_sha(path)}


# ----------------------------------------------------------------------------- reference sets (dev / test)
@dataclass
class Ref:
    role: str            # dev | extra_dev | test | test_source
    ref_id: str          # e.g. validation.jsonl#12 (1-based line) or bipia:email/test.jsonl#3
    subset: str
    text: str


def load_eval_set(path: Path, role: str, text_field: str, label_field: str, gate: Gate, expect_sha: str | None):
    if not path.is_file():
        gate.fail(f"{role.upper()}_MISSING", f"{path} not found")
        return [], {}
    digest = file_sha(path)
    if expect_sha and digest != expect_sha:
        gate.fail(f"{role.upper()}_HASH_MISMATCH", f"{path.name}: sha256 {digest[:12]}... differs from the expected frozen hash")
    try:
        rows = load_records(path)
    except CriticalError as e:
        gate.fail(f"{role.upper()}_UNREADABLE", str(e))
        return [], {}
    bad = [i for i, r in enumerate(rows, 1) if not (isinstance(r, dict) and isinstance(r.get(text_field), str)
                                                     and isinstance(r.get(label_field), int))]
    if bad or not rows:
        gate.fail(f"{role.upper()}_SCHEMA", f"{path.name}: {len(bad)} rows lack a string '{text_field}' or an int '{label_field}'"
                  if rows else f"{path.name}: empty")
    refs = [Ref(role, f"{path.name}#{i}", str(r.get("set", r.get("task", ""))), r[text_field])
            for i, r in enumerate(rows, 1) if isinstance(r, dict) and isinstance(r.get(text_field), str)]
    info = {"path": str(path), "sha256": digest, "rows": len(rows), "text_field": text_field, "label_field": label_field,
            "subsets": dict(collections.Counter(x.subset for x in refs)),
            "labels": dict(collections.Counter(str(r.get(label_field)) for r in rows if isinstance(r, dict)))}
    return refs, info


def load_bipia_test_sources(root: Path, gate: Gate):
    refs, info = [], {"root": str(root), "files": {}}
    b = root / "benchmark"
    for task in ("email", "table", "code"):
        p = b / task / "test.jsonl"
        if not p.is_file():
            gate.fail("BIPIA_SOURCE_MISSING", f"{p} not found")
            continue
        rows = load_records(p)
        info["files"][f"{task}/test.jsonl"] = {"sha256": file_sha(p), "rows": len(rows)}
        for i, r in enumerate(rows, 1):
            ctx = "\n".join(r["context"]) if isinstance(r.get("context"), list) else r.get("context", "")
            refs.append(Ref("test_source", f"bipia:{task}/test.jsonl#{i}", f"bipia_{task}_context", ctx))
    for fname in ("text_attack_test.json", "code_attack_test.json"):
        p = b / fname
        if not p.is_file():
            gate.fail("BIPIA_SOURCE_MISSING", f"{p} not found")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        info["files"][fname] = {"sha256": file_sha(p), "attacks": sum(len(v) for v in data.values())}
        for cat in sorted(data):
            for j, a in enumerate(data[cat], 1):
                refs.append(Ref("test_source", f"bipia:{fname}#{cat}/{j}", "bipia_test_attack", a))
    return refs, info


class RefIndex:
    """Exact, loose and shingle lookups against reference texts."""

    def __init__(self, refs: list):
        self.refs = refs
        self.by_key = collections.defaultdict(list)
        self.by_loose = collections.defaultdict(list)
        self.sh = []
        post = collections.defaultdict(list)
        for i, r in enumerate(refs):
            self.by_key[key(r.text)].append(i)
            lz = loose(r.text)
            if lz:
                self.by_loose[lz].append(i)
            s = shingles(r.text)
            self.sh.append(s)
            for g in s:
                post[g].append(i)
        self.capped = frozenset(g for g, ids in post.items() if len(ids) > POSTING_CAP)
        self.post = {g: ids for g, ids in post.items() if len(ids) <= POSTING_CAP}
        self.sh = [s - self.capped for s in self.sh]
        self.long_keys = [(key(r.text), i) for i, r in enumerate(refs) if len(key(r.text)) >= VERBATIM_MIN]

    def verbatim(self, texts: dict) -> dict:
        """Independent check: every reference whose normalised text has >= VERBATIM_MIN characters is searched
        for verbatim inside every candidate's normalised text. Returns {candidate index: [(kind, ref index, 1.0)]}."""
        order = sorted(texts)
        keys = [key(texts[i]) for i in order]
        starts, pos = [], 0
        for k in keys:
            starts.append(pos)
            pos += len(k) + 1
        big = "\x00".join(keys)
        out = collections.defaultdict(list)
        for rk, ri in self.long_keys:
            p = big.find(rk)
            while p != -1:
                c = order[bisect.bisect_right(starts, p) - 1]
                if not out[c] or out[c][-1][1] != ri:
                    out[c].append(("contains_reference_verbatim", ri, 1.0))
                p = big.find(rk, p + 1)
        return dict(out)

    def match(self, text: str) -> list:
        """Returns [(kind, ref_index, score)] for every reference this text overlaps."""
        hits = {}
        for i in self.by_key.get(key(text), ()):
            hits[i] = ("exact_normalized", 1.0)
        lz = loose(text)
        if lz:
            for i in self.by_loose.get(lz, ()):
                hits.setdefault(i, ("loose_normalized", 1.0))
        # Shingles that occur in more than POSTING_CAP references are boilerplate. They are left out of the
        # shared count AND of both set sizes; counting them only in the sizes would under-state overlap.
        a = shingles(text) - self.capped
        if len(a) >= MIN_SHINGLES:
            shared = collections.Counter(i for g in a for i in self.post.get(g, ()))
            for i, n in shared.items():
                if i in hits:
                    continue
                b = self.sh[i]
                if len(b) < MIN_SHINGLES:
                    continue
                jac = n / (len(a) + len(b) - n)
                if jac >= NEAR_JACCARD:
                    hits[i] = ("near_duplicate", round(jac, 4))
                elif n / len(b) >= CONTAIN:
                    hits[i] = ("contains_reference", round(n / len(b), 4))
                elif n / len(a) >= CONTAIN:
                    hits[i] = ("contained_in_reference", round(n / len(a), 4))
        return [(k, i, s) for i, (k, s) in sorted(hits.items())]


# ----------------------------------------------------------------------------- audit
def audit(args) -> dict:
    gate = Gate()
    report = {"script": "experiments/scripts/audit_injecguard.py", "version": SCRIPT_VERSION,
              "decision_note": "This audit does not approve fine-tuning. It only reports whether a candidate set could be prepared safely.",
              "parameters": {"long_chars_proxy": LONG_CHARS, "near_jaccard": NEAR_JACCARD, "containment": CONTAIN,
                             "min_shingles": MIN_SHINGLES, "posting_cap": POSTING_CAP, "verbatim_min_chars": VERBATIM_MIN}}
    inputs = [p for p in [args.candidates, args.dev, args.frozen_test, *(args.extra_dev or []), args.eval_loader] if p]
    before = {str(p): file_sha(p) for p in inputs if p.is_file()}

    # ---- label mapping evidence
    report["label_mapping"] = {"mapping": {str(k): v for k, v in LABEL_MAPPING.items()},
                               "evidence": verify_label_evidence(args.label_evidence, gate)}

    # ---- candidates: load + schema
    try:
        cand = load_records(args.candidates)
    except CriticalError as e:
        gate.fail("CANDIDATES_UNREADABLE", str(e))
        cand = []
    schema = profile_schema(cand)
    inf_text, inf_label = infer_fields(cand)
    schema.update({"declared_text_field": args.text_field, "declared_label_field": args.label_field,
                   "inferred_text_field": inf_text, "inferred_label_field": inf_label})
    report["candidate_schema"] = schema
    if not cand:
        gate.fail("EMPTY_DATASET", f"{args.candidates.name} has no records")
    elif (inf_text, inf_label) != (args.text_field, args.label_field):
        gate.fail("SCHEMA_MISMATCH", f"declared fields ({args.text_field}, {args.label_field}) differ from inferred "
                  f"({inf_text}, {inf_label})")
    extra = sorted(set(schema["field_types"]) - {args.text_field, args.label_field, args.source_field})
    if extra:
        gate.warn("EXTRA_FIELDS", f"fields not used by the preparation: {extra}")

    stages = collections.OrderedDict()
    stages["loaded"] = len(cand)
    excl = {}                     # index -> list of reasons (first reason decides; all are recorded)
    evid = collections.defaultdict(list)

    def exclude(i, reason, **ev):
        excl.setdefault(i, []).append(reason)
        evid[i].append({"reason": reason, **ev})

    # ---- record validation
    invalid_label, unexpected_label = [], []
    for i, r in enumerate(cand):
        if not isinstance(r, dict):
            exclude(i, "NOT_AN_OBJECT")
            continue
        t, lab = r.get(args.text_field), r.get(args.label_field)
        if not isinstance(t, str):
            exclude(i, "MISSING_TEXT", field=args.text_field)
        elif not t.strip():
            exclude(i, "EMPTY_TEXT")
        if args.label_field not in r or lab is None or isinstance(lab, bool) or not isinstance(lab, int):
            invalid_label.append(i)
            exclude(i, "INVALID_LABEL", value_type=type(lab).__name__)
        elif lab not in LABEL_MAPPING:
            unexpected_label.append(i)
            exclude(i, "UNEXPECTED_LABEL", value=lab)
    if invalid_label:
        gate.fail("INVALID_LABELS", f"{len(invalid_label)} records have a missing or non-integer label")
    if unexpected_label:
        gate.fail("UNEXPECTED_LABELS", f"{len(unexpected_label)} records have labels outside {sorted(LABEL_MAPPING)}")
    n_empty = sum(1 for v in excl.values() if v[0] in ("MISSING_TEXT", "EMPTY_TEXT"))
    if n_empty:
        gate.warn("EMPTY_OR_MISSING_TEXT", f"{n_empty} records excluded for missing or blank text")
    valid = [i for i in range(len(cand)) if i not in excl]
    stages["after_record_validation"] = len(valid)

    def T(i):
        return cand[i][args.text_field]

    def L(i):
        return cand[i][args.label_field]

    def S(i):
        return str(cand[i].get(args.source_field, ""))

    # ---- per-source profile (semantic sanity of labels; structure of text)
    prof = collections.defaultdict(lambda: {"rows": 0, "label_counts": collections.Counter(), "multiline": 0,
                                            "template_prefix": 0, "long_over_proxy": 0, "chars_total": 0})
    for i in valid:
        p = prof[S(i)]
        p["rows"] += 1
        p["label_counts"][str(L(i))] += 1
        p["multiline"] += "\n" in T(i)
        p["template_prefix"] += TEMPLATE_PATTERN.search(T(i)) is not None
        p["long_over_proxy"] += len(T(i)) > LONG_CHARS
        p["chars_total"] += len(T(i))
    report["sources"] = {s: {"rows": v["rows"], "label_counts": dict(v["label_counts"]), "multiline_rows": v["multiline"],
                             "template_prefix_rows": v["template_prefix"], "rows_over_long_proxy": v["long_over_proxy"],
                             "mean_chars": round(v["chars_total"] / v["rows"], 1), "review_note": REVIEW_SOURCES.get(s)}
                         for s, v in sorted(prof.items())}
    long_rows = sum(v["long_over_proxy"] for v in prof.values())
    if long_rows:
        gate.warn("LONG_TEXTS", f"{long_rows} rows exceed {LONG_CHARS} characters (~512 tokens); training would truncate them")
    for s in sorted(prof):
        if s in REVIEW_SOURCES:
            gate.warn("SOURCE_NEEDS_REVIEW", f"{s}: {REVIEW_SOURCES[s]}")

    # ---- target leakage patterns in text (reported, not excluded)
    leak = {}
    srcs = sorted({S(i) for i in valid if len(S(i)) >= 6})
    for name, pat in TARGET_LEAK_PATTERNS.items():
        rx = re.compile(pat, re.I)
        leak[name] = sum(1 for i in valid if rx.search(T(i)))
    src_hits = collections.Counter()
    for i in valid:
        low = T(i).casefold()
        for s in srcs:
            if s.casefold() in low:
                src_hits[s] += 1
    leak["source_name_in_text"] = dict(src_hits)
    report["target_leakage_scan"] = leak
    if any(v for k, v in leak.items() if k != "source_name_in_text") or src_hits:
        gate.warn("POSSIBLE_TARGET_LEAKAGE", "label-like patterns or source names occur in some texts; see target_leakage_scan")

    # ---- duplicates and conflicting labels (on the normalised key)
    groups = collections.defaultdict(list)
    for i in valid:
        groups[key(T(i))].append(i)
    exact_raw = collections.Counter(T(i) for i in valid)
    conflict_groups = [g for g in groups.values() if len({L(i) for i in g}) > 1]
    for g in conflict_groups:
        labels = sorted({L(i) for i in g})
        for i in g:
            exclude(i, "CONFLICTING_LABELS", group_text_sha256=sha(key(T(i))), labels_in_group=labels,
                    group_size=len(g), sources_in_group=sorted({S(j) for j in g}))
    dup_removed = 0
    for g in groups.values():
        if len({L(i) for i in g}) > 1 or len(g) == 1:
            continue
        keep = g[0]
        for i in g[1:]:
            dup_removed += 1
            exclude(i, "DUPLICATE_OF_KEPT_RECORD", kept_record=f"{args.candidates.name}#{keep}",
                    exact_raw_match=T(i) == T(keep), group_text_sha256=sha(key(T(i))))
    report["duplicates"] = {
        "exact_raw_duplicate_groups": sum(1 for c in exact_raw.values() if c > 1),
        "exact_raw_extra_rows": sum(c - 1 for c in exact_raw.values() if c > 1),
        "normalized_groups_with_duplicates": sum(1 for g in groups.values() if len(g) > 1),
        "normalized_extra_rows_removed": dup_removed,
        "conflicting_label_groups": len(conflict_groups),
        "conflicting_label_rows_excluded": sum(len(g) for g in conflict_groups),
    }
    if conflict_groups:
        gate.warn("CONFLICTING_LABELS", f"{len(conflict_groups)} normalised texts carry both labels; all "
                  f"{sum(len(g) for g in conflict_groups)} rows excluded (no label was chosen)")
    if dup_removed:
        gate.warn("DUPLICATES_REMOVED", f"{dup_removed} duplicate rows removed (first occurrence kept)")
    stages["after_dedup_and_conflicts"] = sum(1 for i in valid if i not in excl)

    # near-duplicates inside the candidates (report only: loose key collisions across distinct normalised texts)
    lg = collections.defaultdict(set)
    for i in valid:
        if i not in excl:
            lg[loose(T(i))].add(i)
    near_groups = [g for k, g in lg.items() if k and len(g) > 1]
    near_conf = [g for g in near_groups if len({L(i) for i in g}) > 1]
    report["near_duplicates_within_candidates"] = {
        "method": "loose key (letters and digits only) after normalised de-duplication; report only, not excluded",
        "groups": len(near_groups), "rows_in_groups": sum(len(g) for g in near_groups),
        "groups_with_conflicting_labels": len(near_conf),
        "conflicting_group_hashes": sorted(sha(loose(T(next(iter(g)))))[:16] for g in near_conf)[:50]}
    if near_conf:
        gate.warn("NEAR_DUPLICATE_LABEL_CONFLICTS", f"{len(near_conf)} groups differ only in punctuation/spacing but carry both labels (kept; review)")

    # ---- reference sets
    refs, ref_info = [], {}
    dev_refs, ref_info["dev"] = load_eval_set(args.dev, "dev", args.dev_text_field, args.dev_label_field, gate, args.expect_dev_sha256)
    test_refs, ref_info["frozen_test"] = load_eval_set(args.frozen_test, "test", args.dev_text_field, args.dev_label_field, gate,
                                                       args.expect_test_sha256)
    refs += dev_refs + test_refs
    ref_info["extra_dev"] = []
    for p in args.extra_dev or []:
        r2, info = load_eval_set(p, "extra_dev", args.dev_text_field, args.dev_label_field, gate, None)
        refs += r2
        ref_info["extra_dev"].append(info)
    if args.bipia:
        r3, ref_info["bipia_test_sources"] = load_bipia_test_sources(args.bipia, gate)
        refs += r3
    else:
        gate.warn("BIPIA_SOURCES_NOT_CHECKED", "no --bipia given; BIPIA test contexts and attack pools were not checked for containment")
    ref_info["eval_loader"] = verify_eval_loader(args.eval_loader, args.dev_text_field, args.dev_label_field, gate)
    report["reference_sets"] = ref_info

    # ---- whole-source exclusion: a candidate source that IS the dev set
    dev_sources = collections.Counter()
    if dev_refs:
        dev_keys = {key(r.text) for r in dev_refs}
        by_src = collections.defaultdict(lambda: [0, 0])
        for i in valid:
            by_src[S(i)][0] += 1
            by_src[S(i)][1] += key(T(i)) in dev_keys
        for s, (n, hit) in by_src.items():
            if n >= args.dev_source_min_rows and hit / n >= args.dev_source_share:
                dev_sources[s] = n
        for i in valid:
            if S(i) in dev_sources:
                exclude(i, "SOURCE_IS_DEV_SET", source=S(i), source_rows=dev_sources[S(i)])
    if dev_sources:
        gate.warn("DEV_SOURCE_EXCLUDED", f"whole source(s) excluded because they make up the dev set: {dict(dev_sources)}")

    # ---- overlap with dev / test / test sources
    idx = RefIndex(refs)
    overlap = collections.Counter()
    overlap_sources = collections.defaultdict(collections.Counter)
    verb = idx.verbatim({i: T(i) for i in valid})
    for i in valid:
        hits = idx.match(T(i))
        seen = {j for _, j, _ in hits}
        hits += [h for h in verb.get(i, ()) if h[1] not in seen]
        for kind, j, score in hits:
            r = refs[j]
            reason = {"dev": "OVERLAP_DEV", "extra_dev": "OVERLAP_EXTRA_DEV", "test": "OVERLAP_FROZEN_TEST",
                      "test_source": "OVERLAP_TEST_SOURCE"}[r.role]
            exclude(i, reason, match=kind, score=score, reference=r.ref_id, reference_subset=r.subset,
                    reference_text_sha256=sha(r.text))
            overlap[(reason, kind)] += 1
            overlap_sources[reason][S(i)] += 1
    report["overlap"] = {"matches_by_reason_and_kind": {f"{a}|{b}": n for (a, b), n in sorted(overlap.items())},
                         "candidate_rows_by_reason_and_source": {k: dict(v) for k, v in sorted(overlap_sources.items())}}
    n_ov = {r: sum(1 for i, rs in excl.items() if r in rs) for r in ("OVERLAP_DEV", "OVERLAP_EXTRA_DEV", "OVERLAP_FROZEN_TEST", "OVERLAP_TEST_SOURCE")}
    for r, n in n_ov.items():
        if n:
            gate.warn(r, f"{n} candidate rows overlap {r.split('_', 1)[1].lower().replace('_', ' ')} texts and were excluded")

    kept = [i for i in range(len(cand)) if i not in excl]
    stages["after_leakage_exclusion"] = len(kept)

    # ---- post-exclusion verification (must be zero; otherwise FAIL)
    verb_kept = idx.verbatim({i: T(i) for i in kept})
    residual = sum(1 for i in kept if idx.match(T(i)) or verb_kept.get(i))
    report["post_exclusion_verification"] = {
        "remaining_overlaps": residual,
        "checks": ["exact normalised", "loose normalised", f"word-3-gram Jaccard >= {NEAR_JACCARD}",
                   f"shingle containment >= {CONTAIN} either way", f"verbatim containment of references >= {VERBATIM_MIN} chars"]}
    if residual:
        gate.fail("RESIDUAL_OVERLAP", f"{residual} kept rows still overlap a reference set")
    if not kept:
        gate.fail("EMPTY_CANDIDATE", "no records remain after exclusions")
    else:
        lab = collections.Counter(L(i) for i in kept)
        if set(lab) != set(LABEL_MAPPING):
            gate.fail("CLASS_MISSING", f"candidate labels present: {dict(lab)}")
        elif min(lab.values()) / len(kept) < 0.25:
            gate.warn("CLASS_IMBALANCE", f"label counts {dict(lab)}; minority share {min(lab.values()) / len(kept):.3f}")
        report["candidate_label_counts"] = {LABEL_MAPPING[k]: v for k, v in sorted(lab.items())}
        report["candidate_rows_by_source"] = dict(sorted(collections.Counter(S(i) for i in kept).items()))
    report["stages"] = dict(stages)
    report["exclusions_by_primary_reason"] = dict(sorted(collections.Counter(v[0] for v in excl.values()).items()))
    report["exclusions_any_reason"] = dict(sorted(collections.Counter(r for v in excl.values() for r in set(v)).items()))

    # ---- inputs unchanged?
    after = {p: file_sha(Path(p)) for p in before}
    changed = [p for p in before if before[p] != after[p]]
    if changed:
        gate.fail("INPUT_CHANGED_DURING_RUN", f"{changed}")
    report["inputs_sha256"] = before
    report["status"] = gate.status
    report["critical"] = gate.critical
    report["warnings"] = gate.warnings
    return {"report": report, "cand": cand, "kept": kept, "excl": excl, "evid": evid}


# ----------------------------------------------------------------------------- output safety + writing
def prepare_out(out: Path, overwrite: bool, protected: list) -> None:
    out = out.resolve()
    for p in protected:
        pr = p.resolve()
        base = pr.parent if pr.is_file() or pr.suffix else pr
        if out == base or base in out.parents:
            raise CriticalError(f"refusing to write into {out}: it is inside an input location ({base})")
    if out.exists():
        if not out.is_dir():
            raise CriticalError(f"{out} exists and is not a directory")
        contents = sorted(x.name for x in out.iterdir())
        if contents:
            if not overwrite:
                raise CriticalError(f"{out} is not empty; choose a new --out or pass --overwrite")
            marker = out / MARKER
            if not marker.is_file():
                raise CriticalError(f"{out} was not created by this script (no {MARKER}); not overwriting")
            ours = set(json.loads(marker.read_text())["files"]) | {MARKER}
            foreign = [c for c in contents if c not in ours]
            if foreign:
                raise CriticalError(f"{out} contains files this script did not create: {foreign}; not overwriting")
            for c in contents:
                (out / c).unlink()
    out.mkdir(parents=True, exist_ok=True)


def write_outputs(args, res) -> dict:
    out, rep, cand = args.out, res["report"], res["cand"]
    written = []

    def dump(name, text):
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        written.append(name)

    lines = []
    for i in sorted(res["excl"]):
        r = cand[i] if isinstance(cand[i], dict) else {}
        t = r.get(args.text_field)
        lines.append(json.dumps({"record_id": f"{args.candidates.name}#{i}", "source_index": i,
                                 "source": r.get(args.source_field), "label": r.get(args.label_field),
                                 "text_sha256": sha(t) if isinstance(t, str) else None,
                                 "primary_reason": res["excl"][i][0], "reasons": sorted(set(res["excl"][i])),
                                 "evidence": res["evid"][i]}, sort_keys=True, ensure_ascii=False))
    dump("exclusion_manifest.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    cand_hash = None
    if rep["status"] != "FAIL":
        rows = [json.dumps({"id": f"{args.candidates.stem}:{i:06d}", "text": cand[i][args.text_field],
                            "label": cand[i][args.label_field], "label_name": LABEL_MAPPING[cand[i][args.label_field]],
                            "source": cand[i].get(args.source_field), "text_sha256": sha(cand[i][args.text_field])},
                           ensure_ascii=False, sort_keys=True) for i in res["kept"]]
        dump("candidate_train.jsonl", "\n".join(rows) + "\n")
        cand_hash = file_sha(out / "candidate_train.jsonl")

    dump("audit_report.json", json.dumps(rep, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "script": rep["script"], "version": SCRIPT_VERSION, "status": rep["status"],
        "sources": {"candidates": {"path": str(args.candidates), "sha256": rep["inputs_sha256"].get(str(args.candidates)),
                                   "records": rep["candidate_schema"]["records"]},
                    "reference_sets": rep["reference_sets"]},
        "schema": {"text_field": args.text_field, "label_field": args.label_field, "source_field": args.source_field,
                   "observed": rep["candidate_schema"]},
        "label_mapping": rep["label_mapping"],
        "preprocessing": [
            "Output text is the original string, byte for byte; nothing is rewritten or truncated.",
            "Records with a non-string or blank text field are excluded (EMPTY_TEXT / MISSING_TEXT).",
            "Records with a missing, non-integer or unmapped label make the gate FAIL.",
            "Comparison key: NFKC, casefold, whitespace collapsed, stripped. Duplicates on this key keep the first occurrence.",
            "Texts whose key carries both labels are all excluded; no label is chosen.",
            "A source whose rows make up the dev set is excluded entirely.",
            f"Overlap with dev, extra dev, frozen test or BIPIA test sources (exact key, loose key, word-3-gram Jaccard >= {NEAR_JACCARD}, "
            f"containment >= {CONTAIN}, or verbatim containment of a reference of >= {VERBATIM_MIN} characters) excludes the candidate row.",
            "Order of kept rows = order in the source file. No sampling, no randomness.",
        ],
        "outputs": {"candidate_train.jsonl": cand_hash},
        "approval": "NOT APPROVED FOR FINE-TUNING. This manifest documents a candidate only.",
    }
    for name in ("exclusion_manifest.jsonl", "audit_report.json"):
        manifest["outputs"][name] = file_sha(out / name)
    dump("dataset_manifest.json", json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    dump("AUDIT_SUMMARY.md", summary_md(rep, cand_hash))
    (out / MARKER).write_text(json.dumps({"files": written}, indent=1) + "\n")
    return manifest


def summary_md(rep, cand_hash) -> str:
    L = [f"# InjecGuard data audit: **{rep['status']}**", "",
         "This audit does not approve fine-tuning. It reports whether a candidate set could be prepared safely.", "",
         "## Counts at each stage", "", "| Stage | Rows |", "| --- | --- |"]
    L += [f"| {k.replace('_', ' ')} | {v:,} |" for k, v in rep.get("stages", {}).items()]
    L += ["", "## Exclusions (primary reason)", "", "| Reason | Rows |", "| --- | --- |"]
    L += [f"| {k} | {v:,} |" for k, v in rep.get("exclusions_by_primary_reason", {}).items()]
    d = rep.get("duplicates", {})
    L += ["", "## Duplicates", "", f"- Exact raw duplicate groups: {d.get('exact_raw_duplicate_groups', 0):,} "
          f"({d.get('exact_raw_extra_rows', 0):,} extra rows)",
          f"- Normalised duplicate rows removed: {d.get('normalized_extra_rows_removed', 0):,}",
          f"- Conflicting-label groups: {d.get('conflicting_label_groups', 0)} "
          f"({d.get('conflicting_label_rows_excluded', 0)} rows excluded)"]
    nd = rep.get("near_duplicates_within_candidates", {})
    L += [f"- Near-duplicate groups kept for review: {nd.get('groups', 0):,}; with conflicting labels: {nd.get('groups_with_conflicting_labels', 0)}"]
    L += ["", "## Overlap with evaluation data", ""]
    L += [f"- {k}: {v:,} matches" for k, v in rep.get("overlap", {}).get("matches_by_reason_and_kind", {}).items()] or ["- none"]
    L += [f"- Remaining overlaps after exclusion: {rep.get('post_exclusion_verification', {}).get('remaining_overlaps')}"]
    lm = rep.get("label_mapping", {})
    L += ["", "## Label mapping", "", f"- {lm.get('mapping')}"]
    L += [f"- {e['file']}: {'found' if e['found'] else 'NOT FOUND'} — {e['meaning']}" for e in lm.get("evidence", [])]
    if "candidate_label_counts" in rep:
        L += ["", "## Candidate", "", f"- Label counts: {rep['candidate_label_counts']}",
              f"- candidate_train.jsonl sha256: `{cand_hash}`" if cand_hash else "- Not written (gate FAIL)"]
    L += ["", "## Critical issues", ""] + ([f"- **{c['code']}**: {c['detail']}" for c in rep["critical"]] or ["- none"])
    L += ["", "## Warnings for review", ""] + ([f"- **{w['code']}**: {w['detail']}" for w in rep["warnings"]] or ["- none"])
    return "\n".join(L) + "\n"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--label-evidence", type=Path, help="InjecGuard repo root (eval.py, util.py, train.py)")
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--frozen-test", type=Path, required=True)
    ap.add_argument("--extra-dev", type=Path, action="append")
    ap.add_argument("--bipia", type=Path)
    ap.add_argument("--eval-loader", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true", help="replace outputs ONLY in a directory this script created")
    ap.add_argument("--text-field", default="prompt")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--source-field", default="source")
    ap.add_argument("--dev-text-field", default="text")
    ap.add_argument("--dev-label-field", default="label")
    ap.add_argument("--dev-source-share", type=float, default=0.95,
                    help="a candidate source is treated as the dev set when this share of its rows matches dev texts")
    ap.add_argument("--dev-source-min-rows", type=int, default=20,
                    help="...and it has at least this many rows (smaller matches are handled row by row as OVERLAP_DEV)")
    ap.add_argument("--expect-dev-sha256")
    ap.add_argument("--expect-test-sha256")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    try:
        protected = [args.candidates, args.dev, args.frozen_test, *(args.extra_dev or [])]
        protected += [x for x in (args.bipia, args.label_evidence) if x]
        prepare_out(args.out, args.overwrite, protected)
    except CriticalError as e:
        print(f"OUTPUT REFUSED: {e}", file=sys.stderr)
        return 3
    res = audit(args)
    write_outputs(args, res)
    rep = res["report"]
    print(f"status: {rep['status']}")
    print(json.dumps({"stages": rep.get("stages"), "exclusions": rep.get("exclusions_by_primary_reason"),
                      "critical": [c["code"] for c in rep["critical"]], "warnings": [w["code"] for w in rep["warnings"]]}, indent=1))
    return 2 if rep["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

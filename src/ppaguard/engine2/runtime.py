"""Engine 2 inference: ONNX Runtime + the model's own tokenizer, CPU only.

A model directory contains `model_card.json`, an ONNX file and a `tokenizer.json`. The card records provenance,
label order, file hashes (checked at load), calibration temperature and the recommended WARN/BLOCK thresholds.

Long inputs are not silently truncated: the token sequence is scored in overlapping windows of the model's maximum
length and the highest window probability is used. If the input needs more windows than `max_windows`, only the first
windows are scored and the result says so (`partial=True`), which the policy treats as a rule signal.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import softmax

PROVENANCE = {
    "fine-tuned-ppaguard": "fine-tuned by the PPA Guard training pipeline",
    "trained-from-scratch-ppaguard": "trained from random initialisation by the PPA Guard pipeline (no pretraining)",
    "pretrained-external": "pretrained third-party classifier, used as published",
    "test-only": "TEST ARTIFACT - not a detector",
}


class ModelUnavailable(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    provenance: str
    labels: tuple[str, ...]
    injection_index: int
    max_length: int
    model_file: str = "model.onnx"
    tokenizer_file: str = "tokenizer.json"
    window_stride: int = 256
    max_windows: int = 8
    temperature: float = 1.0
    thresholds: dict = field(default_factory=lambda: {"warn": 0.5, "block": 0.92, "source": "default"})
    files: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ModelCard":
        d = json.loads(path.read_text(encoding="utf-8"))
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k != "extra"}
        known["labels"] = tuple(known["labels"])
        return cls(**known, extra={k: v for k, v in d.items() if k not in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Detection:
    probability: float              # calibrated P(injection), max over windows
    raw_probability: float          # uncalibrated (temperature 1)
    label: str
    model_id: str
    provenance: str
    n_tokens: int
    windows: int
    partial: bool                   # True when more windows were needed than max_windows
    latency_ms: float
    window_probabilities: tuple[float, ...] = ()
    evidence: tuple[tuple[str, float], ...] = ()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_segments(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    if len(parts) >= 2:
        return parts
    words = text.split()
    if len(words) < 8:
        return []
    size = max(3, len(words) // 8)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


class Detector:
    def __init__(self, model_dir: str | Path, threads: int | None = None, verify_hashes: bool = True):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise ModelUnavailable(f"inference dependency missing: {e}") from e
        self.dir = Path(model_dir)
        card_path = self.dir / "model_card.json"
        if not card_path.is_file():
            raise ModelUnavailable(f"no model_card.json in {self.dir}")
        try:
            self.card = ModelCard.load(card_path)
        except Exception as e:
            raise ModelUnavailable(f"invalid model_card.json: {e}") from e
        if self.card.provenance not in PROVENANCE:
            raise ModelUnavailable(f"unknown provenance {self.card.provenance!r}")
        mpath, tpath = self.dir / self.card.model_file, self.dir / self.card.tokenizer_file
        for p in (mpath, tpath):
            if not p.is_file():
                raise ModelUnavailable(f"missing file {p.name}")
        if verify_hashes:
            for name, expected in self.card.files.items():
                actual = _sha256(self.dir / name)
                if actual != expected:
                    raise ModelUnavailable(f"{name} checksum mismatch (expected {expected[:12]}..., got {actual[:12]}...)")
        if not 0 <= self.card.injection_index < len(self.card.labels):
            raise ModelUnavailable("injection_index outside label list")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        try:
            self.session = ort.InferenceSession(str(mpath), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelUnavailable(f"ONNX model could not be loaded: {e}") from e
        self.inputs = {i.name for i in self.session.get_inputs()}
        if "input_ids" not in self.inputs:
            raise ModelUnavailable("ONNX model has no input_ids input")
        out = self.session.get_outputs()[0]
        n_out = out.shape[-1] if out.shape and isinstance(out.shape[-1], int) else None
        if n_out is not None and n_out != len(self.card.labels):
            raise ModelUnavailable(f"model has {n_out} outputs but the card lists {len(self.card.labels)} labels")
        self.tok = Tokenizer.from_file(str(tpath))
        self.tok.no_truncation()
        self.tok.no_padding()
        specials = self.tok.encode("", add_special_tokens=True).ids
        self.prefix, self.suffix = specials[:-1] if len(specials) > 1 else specials, specials[-1:] if len(specials) > 1 else []
        self.body = self.card.max_length - len(self.prefix) - len(self.suffix)
        if self.body < 16:
            raise ModelUnavailable("max_length too small for the tokenizer's special tokens")
        pad = self.tok.token_to_id("[PAD]")
        if pad is None:
            pad = self.tok.token_to_id("<pad>")
        self.pad_id = pad if pad is not None else 0
        self.vocab_size = self.tok.get_vocab_size()
        self.model_id = self.card.model_id
        self.context_signal = None  # Will be set by load_detector if available

    # ------------------------------------------------------------------ core
    def _windows(self, text: str) -> tuple[list[list[int]], int, bool]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not ids:
            return [self.prefix + self.suffix], 0, False
        stride = min(self.card.window_stride, self.body)
        starts = list(range(0, max(1, len(ids) - self.body + stride), stride)) if len(ids) > self.body else [0]
        partial = len(starts) > self.card.max_windows
        starts = starts[: self.card.max_windows]
        return [self.prefix + ids[s:s + self.body] + self.suffix for s in starts], len(ids), partial

    def _logits(self, windows: list[list[int]]) -> np.ndarray:
        width = max(len(w) for w in windows)
        ids = np.full((len(windows), width), self.pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, w in enumerate(windows):
            ids[i, :len(w)] = w
            mask[i, :len(w)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        try:
            out = self.session.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]
        except Exception as e:
            raise InferenceError(f"ONNX inference failed: {e}") from e
        out = np.asarray(out, dtype=np.float64)
        if out.ndim != 2 or out.shape[1] != len(self.card.labels) or not np.isfinite(out).all():
            raise InferenceError(f"unexpected model output shape {out.shape} or non-finite values")
        return out

    def probability(self, text: str) -> tuple[float, float, int, int, bool, tuple[float, ...]]:
        wins, n, partial = self._windows(text)
        logits = self._logits(wins)
        k = self.card.injection_index
        cal = softmax(logits, self.card.temperature)[:, k]
        raw = softmax(logits, 1.0)[:, k]
        return float(cal.max()), float(raw.max()), n, len(wins), partial, tuple(round(float(x), 4) for x in cal)

    def detect(self, text: str, explain: bool = False, max_segments: int = 20) -> Detection:
        if not isinstance(text, str):
            raise InferenceError("input must be a string")
        t0 = time.perf_counter()
        p, raw, n, nw, partial, per = self.probability(text)
        evidence: tuple[tuple[str, float], ...] = ()
        if explain:
            evidence = self.explain(text, p, max_segments)
        label = self.card.labels[self.card.injection_index] if p >= 0.5 else self.card.labels[1 - self.card.injection_index] \
            if len(self.card.labels) == 2 else "n/a"
        return Detection(p, raw, label, self.model_id, self.card.provenance, n, nw, partial,
                         (time.perf_counter() - t0) * 1000, per, evidence)

    def explain(self, text: str, base: float, max_segments: int = 20) -> tuple[tuple[str, float], ...]:
        """Occlusion: how much P(injection) drops when each sentence (or word span) is removed. Positive = the
        segment pushed the score up. Computed with the same model; it is an approximation, not a proof."""
        segs = split_segments(text)
        if len(segs) < 2 or len(segs) > max_segments:
            return ()
        drops = []
        for i, s in enumerate(segs):
            rest = " ".join(segs[:i] + segs[i + 1:])
            drops.append((s, round(base - self.probability(rest)[0], 4)))
        drops.sort(key=lambda x: -x[1])
        return tuple(d for d in drops[:3] if d[1] > 0)

    def status(self) -> dict:
        return {"model_id": self.model_id, "provenance": self.card.provenance, "provenance_text": PROVENANCE[self.card.provenance],
                "max_length": self.card.max_length, "temperature": self.card.temperature, "thresholds": self.card.thresholds,
                "vocab_size": self.vocab_size, "files_verified": bool(self.card.files),
                "status": self.card.extra.get("status", "unvalidated"), "limitations": self.card.extra.get("limitations")}

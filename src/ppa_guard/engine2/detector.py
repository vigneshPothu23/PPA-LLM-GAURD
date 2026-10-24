"""Engine 2 -- prompt-injection detector using ProtectAI's pretrained DeBERTa checkpoint.

Model   : protectai/deberta-v3-base-prompt-injection-v2  (Apache-2.0, not gated)
Revision: 89b085cd330414d3e7d9dd787870f315957e1e9f  -- the revision pinned by ProtectAI's own
          LLM Guard 0.3.16 (llm_guard/input_scanners/prompt_injection.py, V2_MODEL)
Runtime : ONNX Runtime on CPU, using the ONNX export ProtectAI ships in the model repo (onnx/).
          No torch, no training, no fine-tuning.

Decision rule mirrored from LLM Guard (MatchType.FULL):
    score = P(INJECTION), rounded to 2 decimals
    flagged as injection when score > 0.92  (strict)
    input truncated to 512 tokens
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HF_REPO = "protectai/deberta-v3-base-prompt-injection-v2"
HF_REVISION = "89b085cd330414d3e7d9dd787870f315957e1e9f"
ONNX_FILES = ("onnx/model.onnx", "onnx/tokenizer.json", "onnx/config.json")
MAX_TOKENS = 512
OFFICIAL_THRESHOLD = 0.92      # LLM Guard default (score > threshold)
MODEL_CARD_THRESHOLD = 0.5     # argmax of the 2-class softmax (model card pipeline)


@dataclass(frozen=True)
class Detection:
    """Raw Engine 2 output. Decisions are made in policy.py, not here."""

    score: float                     # P(INJECTION), rounded to 2 decimals like LLM Guard
    raw_score: float                 # unrounded softmax probability
    label: str                       # model's argmax label: "INJECTION" or "SAFE"
    model_id: str
    n_tokens: int
    truncated: bool
    latency_ms: float
    # sentence-occlusion attribution: (sentence, score drop when removed), strongest first
    evidence: tuple[tuple[str, float], ...] = field(default_factory=tuple)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT.split(text)) if s]


class DebertaDetector:
    """Loads the ONNX export of the checkpoint from a local folder that mirrors the repo layout."""

    backend = "deberta-v3-base-prompt-injection-v2 (ONNX)"

    def __init__(self, model_dir: str | Path, revision: str = HF_REVISION, threads: int | None = None,
                 expected_sha256: str | None = None, model_id: str | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_path = model_dir / "onnx" / "model.onnx"
        if expected_sha256:
            actual = _sha256(onnx_path)
            if actual != expected_sha256:
                raise RuntimeError(f"model.onnx checksum mismatch: expected {expected_sha256}, got {actual}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        tok_path = str(model_dir / "onnx" / "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)          # truncating copy (what the model sees)
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self._counter = Tokenizer.from_file(tok_path)           # non-truncating copy (to report length)
        self._counter.no_padding()
        self._counter.no_truncation()
        import json

        cfg = json.loads((model_dir / "onnx" / "config.json").read_text())
        id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.injection_index = next(k for k, v in id2label.items() if v.upper() == "INJECTION")
        self.model_id = model_id or f"{HF_REPO}@{revision[:12]}"

    # -- core ---------------------------------------------------------------
    def _probability(self, text: str) -> tuple[float, int, bool]:
        enc = self.tokenizer.encode(text)
        n_full = len(self._counter.encode(text).ids)
        feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
                 "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        logits = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
        p = float(_softmax(logits)[0, self.injection_index])
        return p, n_full, n_full > MAX_TOKENS

    def detect(self, text: str, explain: bool = False, max_sentences: int = 12) -> Detection:
        import time

        t0 = time.perf_counter()
        p, n_tokens, truncated = self._probability(text)
        evidence: tuple[tuple[str, float], ...] = ()
        if explain:
            evidence = self._occlusion(text, p, max_sentences)
        return Detection(round(p, 2), p, "INJECTION" if p >= 0.5 else "SAFE", self.model_id,
                         n_tokens, truncated, (time.perf_counter() - t0) * 1000, evidence)

    def _occlusion(self, text: str, base: float, max_sentences: int) -> tuple[tuple[str, float], ...]:
        """Score drop when each sentence is removed (approximate attribution; extra forward passes)."""
        sents = split_sentences(text)
        if len(sents) < 2 or len(sents) > max_sentences:
            return ()
        drops = []
        for i, s in enumerate(sents):
            rest = " ".join(sents[:i] + sents[i + 1:])
            drops.append((s, round(base - self._probability(rest)[0], 3)))
        return tuple(sorted(drops, key=lambda x: -x[1])[:3])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(cache_dir: str | Path | None = None, revision: str = HF_REVISION) -> Path:
    """Download only the ONNX files needed (~750 MB) at the pinned revision. Free; no account needed."""
    from huggingface_hub import hf_hub_download

    for f in ONNX_FILES:
        path = hf_hub_download(HF_REPO, f, revision=revision, cache_dir=cache_dir)
    return Path(path).parent.parent  # .../snapshots/<revision>/

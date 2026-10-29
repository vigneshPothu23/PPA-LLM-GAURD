"""Optional sentence embeddings for the off-topic feature.

A common property of an injected span is that it is *off-topic*: unrelated to the rest of the document it
was planted in. We measure this with cosine distance between a segment's embedding and the embedding of
the surrounding document (the mean of the other segments).

The embedding model is sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), run through ONNX Runtime on
CPU with mean pooling and L2 normalisation - the standard sentence-transformers recipe. It is OPTIONAL:
if the model files are absent, the guard runs without the off-topic feature. Whether it earns its place
is decided by the ablation in the evaluation (docs/v5/CONTEXT_GUARD.md), not assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_dir: str | Path, max_length: int = 256, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dir = Path(model_dir)
        model_file = self.dir / "onnx" / "model.onnx"
        if not model_file.is_file():
            model_file = self.dir / "model.onnx"
        tok_file = self.dir / "tokenizer.json"
        if not model_file.is_file() or not tok_file.is_file():
            raise FileNotFoundError(f"embedding model not found under {self.dir}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_file), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(str(tok_file))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.max_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float64)
        enc = [self.tok.encode(t).ids[: self.max_length] or [0] for t in texts]
        width = max(len(e) for e in enc)
        ids = np.zeros((len(enc), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(enc):
            ids[i, : len(e)] = e
            mask[i, : len(e)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        out = self.session.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]
        out = np.asarray(out, dtype=np.float64)  # (n, seq, 384)
        m = mask[:, :, None].astype(np.float64)
        pooled = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norm, 1e-9, None)


def offtopic_scores(emb: np.ndarray) -> np.ndarray:
    """For each row, 1 - cosine(row, mean of the OTHER rows). Higher = more off-topic. Needs >= 3 rows."""
    n = len(emb)
    if n < 3:
        return np.zeros(n)
    total = emb.sum(axis=0)
    scores = np.zeros(n)
    for i in range(n):
        others = (total - emb[i]) / (n - 1)
        others = others / np.clip(np.linalg.norm(others), 1e-9, None)
        scores[i] = 1.0 - float(emb[i] @ others)
    return scores

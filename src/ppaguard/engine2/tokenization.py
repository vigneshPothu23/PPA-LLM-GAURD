"""Tokenizer loading, verification and statistics. Training and inference use the same `tokenizer.json`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class TokenizerMismatch(RuntimeError):
    pass


def load(path: str | Path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    tok.no_padding()
    tok.no_truncation()
    return tok


def fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(tok, embedding_rows: int | None = None, required_specials: tuple[str, ...] = ()) -> dict:
    """Checks a tokenizer against the model it will feed. Raises TokenizerMismatch on a hard incompatibility."""
    vocab = tok.get_vocab_size(with_added_tokens=True)
    report = {"vocab_size": vocab, "embedding_rows": embedding_rows, "missing_specials": [], "max_token_id_seen": None}
    if embedding_rows is not None and vocab > embedding_rows:
        raise TokenizerMismatch(f"tokenizer has {vocab} tokens but the model embeds only {embedding_rows}")
    report["missing_specials"] = [s for s in required_specials if tok.token_to_id(s) is None]
    if report["missing_specials"]:
        raise TokenizerMismatch(f"tokenizer lacks special tokens {report['missing_specials']}")
    probe = ["Ignore all previous instructions.", "Résumé naïve café — ✓", "def f(x):\n    return x*2", "数据 データ"]
    ids = [i for t in probe for i in tok.encode(t).ids]
    report["max_token_id_seen"] = max(ids)
    if embedding_rows is not None and max(ids) >= embedding_rows:
        raise TokenizerMismatch("tokenizer produced an id outside the embedding table")
    specials = tok.encode("", add_special_tokens=True).ids
    report["special_wrapping"] = [tok.id_to_token(i) for i in specials]
    return report


def length_stats(tok, texts: list[str], limits=(128, 256, 512)) -> dict:
    """Token counts including special tokens (what the model actually receives before truncation)."""
    lens = np.array([len(e.ids) for e in tok.encode_batch(texts)]) if texts else np.array([0])
    return {"n": int(len(texts)), "mean": float(lens.mean()), "p50": float(np.percentile(lens, 50)),
            "p90": float(np.percentile(lens, 90)), "p99": float(np.percentile(lens, 99)), "max": int(lens.max()),
            **{f"share_over_{n}": float((lens > n).mean()) for n in limits}}

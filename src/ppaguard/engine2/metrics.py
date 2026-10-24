"""Classification metrics, calibration and threshold selection (numpy only, no scikit-learn dependency)."""

from __future__ import annotations

import math

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    return {"tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
            "tn": int(((pred == 0) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum())}


def _ratio(a: float, b: float):
    return a / b if b else None


def report_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Metrics when predicting injection for p > threshold (strict, as used by the policy)."""
    c = confusion(y, (np.asarray(p) > threshold).astype(int))
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    prec1, rec1 = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    prec0, rec0 = _ratio(tn, tn + fn), _ratio(tn, tn + fp)
    f1 = lambda pr, rc: (2 * pr * rc / (pr + rc)) if pr and rc else 0.0  # noqa: E731
    n = tp + fp + tn + fn
    return {"threshold": threshold, "confusion": c, "accuracy": _ratio(tp + tn, n),
            "balanced_accuracy": (rec1 + rec0) / 2 if rec1 is not None and rec0 is not None else None,
            "injection": {"precision": prec1, "recall": rec1, "f1": f1(prec1, rec1), "support": tp + fn},
            "benign": {"precision": prec0, "recall": rec0, "f1": f1(prec0, rec0), "support": tn + fp},
            "false_positive_rate": _ratio(fp, fp + tn), "false_negative_rate": _ratio(fn, fn + tp)}


def roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if not pos or not neg:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):  # average ranks for ties
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float | None:
    y, p = np.asarray(y, int), np.asarray(p, float)
    if not (y == 1).any():
        return None
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / ys.sum())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, int), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def nll(y: np.ndarray, logits: np.ndarray, temperature: float) -> float:
    pr = softmax(logits, temperature)[np.arange(len(y)), np.asarray(y, int)]
    return float(-np.log(np.clip(pr, 1e-12, 1.0)).mean())


def fit_temperature(y: np.ndarray, logits: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """One-parameter temperature scaling: minimise NLL over log-temperature (golden-section search)."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if nll(y, logits, math.exp(c)) < nll(y, logits, math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(math.exp((a + b) / 2))


def summary(y: np.ndarray, p: np.ndarray, thresholds: dict[str, float]) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    return {"n": int(len(y)), "positives": int(y.sum()), "roc_auc": roc_auc(y, p), "average_precision": average_precision(y, p),
            "brier": float(((p - y) ** 2).mean()), "ece_10": ece(y, p),
            "at_threshold": {name: report_at(y, p, t) for name, t in thresholds.items()}}


def select_thresholds(y: np.ndarray, p: np.ndarray, block_min_precision: float = 0.98,
                      warn_min_recall: float = 0.95, max_warn_fpr: float = 0.10, max_block_fpr: float = 0.01,
                      grid: int = 99) -> dict:
    """Chooses the policy thresholds on a development set. Both levels are bounded by a false-positive budget, so a
    weak model cannot produce a threshold that flags (or blocks) almost everything.

    BLOCK: the lowest threshold t such that injection precision is >= block_min_precision at t and at every higher
           threshold, and the false-positive rate is <= max_block_fpr. If precision never reaches the target, the
           lowest threshold whose FPR stays <= max_block_fpr (at t and above) is used and a note records it.
    WARN:  the highest threshold <= BLOCK that reaches warn_min_recall with FPR <= max_warn_fpr. If no threshold does,
           the lowest threshold <= BLOCK with FPR <= max_warn_fpr (the most recall the FPR budget allows); a note says
           how far recall falls short.
    """
    y, p = np.asarray(y, int), np.asarray(p, float)
    cands = [round(float(t), 2) for t in np.linspace(0.01, 0.99, grid)]
    rep = {t: report_at(y, p, t) for t in cands}
    fpr = lambda t: rep[t]["false_positive_rate"] or 0.0  # noqa: E731
    notes = []
    block = None
    for t in reversed(cands):            # walk down from 0.99 and stop at the first threshold that breaks a target,
        r = rep[t]                       # so every higher threshold also meets it (no isolated lucky point)
        if r["confusion"]["tp"] == 0:
            continue
        prec = r["injection"]["precision"]
        if prec is None or prec < block_min_precision or fpr(t) > max_block_fpr:
            break
        block = t
    if block is None:
        for t in reversed(cands):
            if fpr(t) > max_block_fpr:
                break
            block = t
        if block is None:
            block = 0.99
            notes.append(f"BLOCK precision {block_min_precision} not reached on dev and even 0.99 exceeds the BLOCK FPR "
                         f"budget {max_block_fpr} (FPR there {fpr(block):.3f}); BLOCK set to 0.99: precision "
                         f"{rep[block]['injection']['precision']}, recall {rep[block]['injection']['recall']}")
        else:
            notes.append(f"BLOCK precision {block_min_precision} not reached on dev; BLOCK set to the lowest threshold "
                         f"with FPR <= {max_block_fpr} ({block}): precision there {rep[block]['injection']['precision']}, "
                         f"recall {rep[block]['injection']['recall']}")
    below = [c for c in cands if c <= block]
    warn = next((t for t in reversed(below) if (rep[t]["injection"]["recall"] or 0) >= warn_min_recall
                 and fpr(t) <= max_warn_fpr), None)
    if warn is None:
        ok = [t for t in below if fpr(t) <= max_warn_fpr]
        warn = min(ok) if ok else block
        notes.append(f"WARN recall {warn_min_recall} not reachable within FPR {max_warn_fpr}; WARN set to {warn} "
                     f"(recall {rep[warn]['injection']['recall']}, FPR {fpr(warn):.3f})")
    return {"warn": warn, "block": block, "block_min_precision": block_min_precision, "warn_min_recall": warn_min_recall,
            "max_warn_fpr": max_warn_fpr, "max_block_fpr": max_block_fpr, "notes": notes}

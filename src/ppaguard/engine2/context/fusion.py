"""A small, dependency-light logistic-regression fusion layer (numpy only).

The clean inference environment has numpy, onnxruntime and tokenizers but not scikit-learn, so the fusion
model is a plain L2-regularised logistic regression trained with batch gradient descent. It is tiny
(a handful of features), so this is more than adequate and keeps the runtime free of new dependencies.

The learned weights are stored as JSON and shipped with the guard artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class LogisticModel:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "LogisticModel":
        return cls(**json.loads(Path(path).read_text()))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        Xs = (X - mean) / std
        z = Xs @ np.asarray(self.weights) + self.bias
        return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: np.ndarray, y: np.ndarray, feature_names: list[str], *,
                 l2: float = 1.0, lr: float = 0.5, epochs: int = 4000,
                 class_weight: bool = True, seed: int = 0) -> LogisticModel:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    if class_weight:
        pos = max(1.0, y.sum())
        neg = max(1.0, n - y.sum())
        sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    else:
        sw = np.ones(n)
    sw = sw / sw.mean()
    for _ in range(epochs):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * sw
        gw = Xs.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return LogisticModel(feature_names, mean.tolist(), std.tolist(), w.tolist(), float(b))

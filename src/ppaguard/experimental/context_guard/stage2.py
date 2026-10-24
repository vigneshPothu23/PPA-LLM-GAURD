"""Round 2 (EXPERIMENTAL): a document-aware second stage on top of the context guard.

Round 1 found two regimes. The segment-level guard wins when a short instruction is hidden inside a larger benign
document, and the whole-document detector wins when the attacker wrote the entire message. A plain maximum of the
two failed (their scores live on different scales). Stage 2 instead learns, on development data that contains both
regimes, how much to trust each signal given the document's structure.

Document features (all from one segmentation pass plus one whole-document detector call):
  z_guard        logit of the segment guard's document score (all features)
  z_guard_nooff  logit of the segment guard's score with the off-topic feature removed
  z_wholedoc     logit of the Engine 2 detector on the whole document
  log_nseg       log(1 + number of segments)
  has_table      document contains table cells
  has_code       document contains code lines or comments
  z_wholedoc_x_lognseg, z_guard_x_lognseg   interactions, so the weight of each signal can depend on length
A small L2 logistic regression (fusion.fit_logistic) maps these to P(injection).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fusion import LogisticModel, fit_logistic

FEATURES = ["z_guard", "z_guard_nooff", "z_wholedoc", "log_nseg", "has_table", "has_code",
            "z_wholedoc_x_lognseg", "z_guard_x_lognseg"]
_EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1 - _EPS)
    return float(np.log(p / (1 - p)))


def doc_features(guard_p: float, guard_nooff_p: float, wholedoc_p: float, kinds: list[str]) -> list[float]:
    zg, zn, zw = _logit(guard_p), _logit(guard_nooff_p), _logit(wholedoc_p)
    ln = float(np.log1p(len(kinds)))
    has_table = 1.0 if "table_cell" in kinds else 0.0
    has_code = 1.0 if any(k in ("code", "code_comment") for k in kinds) else 0.0
    return [zg, zn, zw, ln, has_table, has_code, zw * ln, zg * ln]


def fit_stage2(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> LogisticModel:
    return fit_logistic(np.asarray(X), np.asarray(y), FEATURES, l2=l2, seed=0)


class TwoStageGuard:
    """Live scorer: segment guard (two stage-1 views) + whole-document detector + stage-2 fusion."""

    def __init__(self, guard, stage1_full: LogisticModel, stage1_nooff: LogisticModel, stage2: LogisticModel,
                 offtopic_col: int):
        self.guard = guard            # a ContextGuard with detector + embedder (used for segment features)
        self.s1_full, self.s1_nooff, self.s2 = stage1_full, stage1_nooff, stage2
        self.off_col = offtopic_col

    def components(self, text: str) -> dict:
        segs, X, *_ = self.guard.segment_matrix(text)
        if not segs:
            g = gn = 0.0
            kinds: list[str] = []
        else:
            g = float(self.s1_full.predict_proba(X).max())
            Xn = X.copy()
            Xn[:, self.off_col] = 0.0
            gn = float(self.s1_nooff.predict_proba(Xn).max())
            kinds = [s.kind for s in segs]
        w = float(self.guard.detector.probability(text)[0])
        return {"guard": g, "guard_nooff": gn, "wholedoc": w, "kinds": kinds}

    def probability(self, text: str) -> float:
        c = self.components(text)
        f = doc_features(c["guard"], c["guard_nooff"], c["wholedoc"], c["kinds"])
        return float(self.s2.predict_proba(np.asarray([f]))[0])

    @staticmethod
    def _m(d: dict) -> LogisticModel:
        return LogisticModel(d["feature_names"], d["mean"], d["std"], d["weights"], d["bias"])

    @classmethod
    def load(cls, path: str | Path, guard) -> "TwoStageGuard":
        a = json.loads(Path(path).read_text())
        return cls(guard, cls._m(a["stage1_full"]), cls._m(a["stage1_nooff"]), cls._m(a["stage2"]),
                   a["offtopic_col"])

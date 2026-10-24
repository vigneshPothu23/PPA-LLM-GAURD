"""ContextGuard: experimental context-aware indirect-injection detector.

Pipeline for one untrusted document:
  1. segment it (segment.py) into typed segments;
  2. for each segment, build a feature vector:
       - lexical cues (cues.py): AI-addressing, override framing, directive verbs, imperative start,
         delimiter forgery, url/email, second-person;
       - localized detector score: the existing Engine 2 detector run on the SEGMENT (not the whole
         document), so a short injection is not diluted. To stay cheap, the detector is run only on
         segments that have any lexical cue, plus the single highest-cue segment; other segments get 0;
       - off-topic score (optional, embed.py): how unrelated the segment is to the rest of the document;
       - kind weight: table cells and code comments are slightly up-weighted as injection carriers.
  3. a logistic-regression fusion (fusion.py) maps the feature vector to a per-segment injection
     probability;
  4. document risk = max over segments (an injection anywhere is a problem); the contributing segment is
     returned as evidence.

Training (fit) and thresholds are done on the development set only. Everything is isolated from the
shipped app; see docs/v5/CONTEXT_GUARD.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cues
from .fusion import LogisticModel, fit_logistic
from .segment import Segment, segment

_KIND_WEIGHT = {"table_cell": 1.0, "code_comment": 1.0, "sentence": 0.0, "line": 0.5, "code": -0.5}


@dataclass
class SegmentScore:
    text: str
    kind: str
    probability: float
    detector_score: float
    offtopic: float
    cues: dict


@dataclass
class GuardResult:
    probability: float            # document-level P(injection)
    top_segment: str
    top_kind: str
    n_segments: int
    segments: list[SegmentScore]


class ContextGuard:
    #: feature order used for fusion; kept explicit so artifacts are reproducible
    FEATURES = cues.FEATURE_NAMES + ["detector_score", "offtopic", "kind_weight"]
    #: upper bound on localised detector calls per document (CPU-cost guard)
    MAX_DETECTOR_CALLS = 16

    def __init__(self, detector=None, embedder=None, model: LogisticModel | None = None,
                 use_offtopic: bool = True, use_detector: bool = True):
        self.detector = detector
        self.embedder = embedder
        self.model = model
        self.use_offtopic = use_offtopic and embedder is not None
        self.use_detector = use_detector and detector is not None

    # -------------------------------------------------------------- features
    def _segment_features(self, segs: list[Segment]) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
        texts = [s.text for s in segs]
        cue_rows = [cues.cue_vector(t) for t in texts]

        # localized detector scores (only on segments with a cue, for speed)
        det = np.zeros(len(segs))
        if self.use_detector:
            # Localised detector: run on every segment carrying a lexical cue, plus the longest segment
            # (a safety net for an injected paragraph whose vocabulary dodges the cue patterns). Capped at
            # MAX_DETECTOR_CALLS per document to keep CPU cost bounded; remaining segments keep score 0.
            cue_mass = [sum(cv.values()) for cv in cue_rows]
            run_idx = {i for i, m in enumerate(cue_mass) if m > 0}
            lengths = [len(t) for t in texts]
            run_idx.add(int(np.argmax(lengths)))
            ordered = sorted(run_idx, key=lambda i: -(cue_mass[i] + lengths[i] / 1000.0))
            for i in ordered[: self.MAX_DETECTOR_CALLS]:
                try:
                    det[i] = self.detector.probability(texts[i])[0]
                except Exception:
                    det[i] = 0.0

        # off-topic
        off = np.zeros(len(segs))
        if self.use_offtopic and len(segs) >= 3:
            try:
                from .embed import offtopic_scores
                emb = self.embedder.encode(texts)
                off = offtopic_scores(emb)
            except Exception:
                off = np.zeros(len(segs))

        rows = []
        for i, s in enumerate(segs):
            cv = cue_rows[i]
            row = [cv[name] for name in cues.FEATURE_NAMES]
            row += [det[i], off[i], _KIND_WEIGHT.get(s.kind, 0.0)]
            rows.append(row)
        return np.asarray(rows, dtype=np.float64), cue_rows, det, off

    def segment_matrix(self, text: str) -> tuple[list[Segment], np.ndarray, list[dict], np.ndarray, np.ndarray]:
        segs = segment(text)
        if not segs:
            return [], np.zeros((0, len(self.FEATURES))), [], np.zeros(0), np.zeros(0)
        X, cue_rows, det, off = self._segment_features(segs)
        return segs, X, cue_rows, det, off

    # -------------------------------------------------------------- scoring
    def score(self, text: str) -> GuardResult:
        if self.model is None:
            raise RuntimeError("ContextGuard has no fusion model; call fit() or load a model")
        segs, X, cue_rows, det, off = self.segment_matrix(text)
        if not segs:
            return GuardResult(0.0, "", "", 0, [])
        p = self.model.predict_proba(X)
        scores = [SegmentScore(s.text, s.kind, float(p[i]), float(det[i]), float(off[i]), cue_rows[i])
                  for i, s in enumerate(segs)]
        top = int(np.argmax(p))
        return GuardResult(float(p[top]), segs[top].text, segs[top].kind, len(segs), scores)

    def probability(self, text: str) -> float:
        return self.score(text).probability

    # -------------------------------------------------------------- training
    def fit(self, texts: list[str], labels: list[int], *, l2: float = 1.0, seed: int = 0) -> LogisticModel:
        """Fit the fusion LR on DOCUMENT-level labels, using the max-cue/detector segment per document as
        the training instance (multiple-instance style: a positive document contributes its most
        injection-like segment; a negative contributes the same, forcing the model to push benign
        high-cue segments down)."""
        rows, ys = [], []
        for text, y in zip(texts, labels):
            segs, X, _, det, _ = self.segment_matrix(text)
            if not segs:
                continue
            # pick the representative segment: highest (detector + cue mass)
            cue_mass = X[:, : len(cues.FEATURE_NAMES)].sum(axis=1)
            rep = int(np.argmax(cue_mass + X[:, len(cues.FEATURE_NAMES)]))  # + detector_score column
            rows.append(X[rep])
            ys.append(int(y))
            # for positive docs also add the top-2 segments to give more positive signal
            if y == 1 and len(segs) > 1:
                order = np.argsort(-(cue_mass + X[:, len(cues.FEATURE_NAMES)]))
                for j in order[1:2]:
                    rows.append(X[j])
                    ys.append(1)
        X = np.asarray(rows)
        y = np.asarray(ys)
        self.model = fit_logistic(X, y, self.FEATURES, l2=l2, seed=seed)
        return self.model

    def fit_segments(self, texts: list[str], labels: list[int], payloads: list[str], *,
                     l2: float = 1.0, seed: int = 0) -> LogisticModel:
        """Segment-level supervision (preferred when the injected payload is known).

        For an injected document, segments that overlap the known payload are positive and the remaining
        segments are negative - this teaches the model directly that benign segments of an injected
        document are not themselves injections. For a benign document, every segment is negative. This
        matches the max-over-segments scoring used at inference and gives interpretable weights.
        """
        import re as _re

        def overlaps(seg_text: str, payload: str) -> bool:
            s = _re.sub(r"\s+", " ", seg_text.strip().lower())
            p = _re.sub(r"\s+", " ", payload.strip().lower())
            if not s or not p:
                return False
            return s in p or p in s or (len(s) > 12 and s[:40] in p)

        rows, ys = [], []
        for text, y, payload in zip(texts, labels, payloads):
            segs, X, _, _, _ = self.segment_matrix(text)
            if not segs:
                continue
            for i, s in enumerate(segs):
                if y == 1 and payload and overlaps(s.text, payload):
                    rows.append(X[i]); ys.append(1)
                else:
                    rows.append(X[i]); ys.append(0)
        X = np.asarray(rows)
        y = np.asarray(ys)
        self.model = fit_logistic(X, y, self.FEATURES, l2=l2, seed=seed)
        return self.model

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save")
        meta = {
            "features": self.FEATURES,
            "use_offtopic": self.use_offtopic,
            "use_detector": self.use_detector,
            "model": {"feature_names": self.model.feature_names, "mean": self.model.mean,
                      "std": self.model.std, "weights": self.model.weights, "bias": self.model.bias},
        }
        Path(path).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, detector=None, embedder=None) -> "ContextGuard":
        meta = json.loads(Path(path).read_text())
        m = meta["model"]
        model = LogisticModel(m["feature_names"], m["mean"], m["std"], m["weights"], m["bias"])
        return cls(detector=detector, embedder=embedder, model=model,
                   use_offtopic=meta.get("use_offtopic", False),
                   use_detector=meta.get("use_detector", True))

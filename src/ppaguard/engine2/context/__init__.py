"""Engine 2b - context signal for instructions hidden inside documents (active since the final submission build).

Why it exists (measured, see docs/v5/FINAL_EVALUATION.md): a whole-document classifier dilutes a short instruction
hidden in a long README, e-mail, table or log (a README carrying "ignore prior instructions" in an HTML comment scored
0.0012 as a whole, 1.0000 as a line). Scoring every line, however, also fires on benign lines such as "please ignore my
earlier e-mail". The context guard therefore scores *segments* with three kinds of evidence - lexical cues that separate
instructions addressed to an AI or forging a boundary from ordinary text, the Engine 2 detector run on the segment, and
how off-topic the segment is in its document (MiniLM-L6 sentence embeddings) - fused by a small logistic regression
(10 weights, fitted on development data in round 1; artifact `artifacts/context_guard/guard_full.json`).

In the policy it is a WARN-only signal: above its threshold the input is held for analyst review; it never BLOCKs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .guard import ContextGuard, GuardResult, SegmentScore
from .segment import Segment, segment

__all__ = ["ContextGuard", "ContextSignal", "ContextResult", "GuardResult", "SegmentScore", "Segment", "segment"]


@dataclass(frozen=True)
class ContextResult:
    probability: float
    top_segment: str
    top_kind: str
    n_segments: int


class ContextSignal:
    """Runtime wrapper used by the security workflow. `detector` is the loaded Engine 2 detector (shared)."""

    def __init__(self, artifact: str | Path, detector, embedder):
        self.artifact = Path(artifact)
        self.guard = ContextGuard.load(self.artifact, detector=detector, embedder=embedder)
        if not (self.guard.use_detector and self.guard.use_offtopic):
            raise ValueError("context guard artifact must use both the detector and the off-topic feature")
        self.model_id = "context-guard-full (round 1) + sentence-transformers/all-MiniLM-L6-v2@1110a243"

    def score(self, text: str) -> ContextResult:
        r = self.guard.score(text)
        return ContextResult(r.probability, r.top_segment, r.top_kind, r.n_segments)

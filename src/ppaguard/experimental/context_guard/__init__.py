"""Experimental context-aware indirect-injection detector (UNVALIDATED).

Not a production control and not wired into the default workflow. See docs/v5/CONTEXT_GUARD.md for the
threat model, method, evaluation and limitations.
"""
from .guard import ContextGuard, GuardResult, SegmentScore
from .segment import Segment, segment

__all__ = ["ContextGuard", "GuardResult", "SegmentScore", "Segment", "segment"]

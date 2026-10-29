"""Context-guard research package (rounds 1-2). The guard itself moved to `ppaguard.engine2.context` when it became
part of Engine 2 in the final build; the modules here re-export it for the experiment scripts. What remains
experimental and NOT used by the application: `stage2.py` (round-2 two-stage fusion, failed its pre-registered test)
and `policy_bridge.py` (round-1 adapter). See docs/v5/CONTEXT_GUARD.md and docs/v5/FINAL_EVALUATION.md.
"""
from ...engine2.context import ContextGuard, GuardResult, Segment, SegmentScore, segment

__all__ = ["ContextGuard", "GuardResult", "SegmentScore", "Segment", "segment"]

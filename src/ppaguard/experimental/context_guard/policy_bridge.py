"""Map the experimental context guard onto the EXISTING ALLOW / WARN / BLOCK policy (unchanged).

    result = assess(text, guard)        # -> ppaguard.policy.PolicyResult

The guard's document probability is passed to `ppaguard.policy.engine.evaluate` exactly as an Engine 2
probability would be, together with Engine 1's structural forgery findings (`find_forgery`). All existing
policy behaviour is kept: input validation, hidden-character and forgery rules, fail-closed on errors,
"max severity wins", and a release can never override a BLOCK.

Conservative by default: on the development set the guard did not reach the policy's BLOCK precision
target (0.98), so the guard alone is NOT allowed to BLOCK: its score is clamped at the BLOCK threshold and a
high guard score produces a WARN that is HELD for analyst review. Structural rules
(input validation, boundary forgery configured to BLOCK) can still BLOCK as before. Pass
`allow_guard_block=True` only with evidence that the guard meets the BLOCK target on your own data.

Not imported by the default workflow, SDK or UI.
"""
from __future__ import annotations

from ...engine1 import find_forgery
from ...policy.engine import Decision, PolicyConfig, PolicyResult, evaluate

#: thresholds selected on ctx_dev (docs/v5/evidence/context_guard/comparison.json, guard_full)
DEFAULT_WARN = 0.97
CONSERVATIVE_BLOCK = 0.999


def config(warn_threshold: float = DEFAULT_WARN, block_threshold: float = CONSERVATIVE_BLOCK,
           max_chars: int = 20000, warn_handling: str = "hold") -> PolicyConfig:
    return PolicyConfig(warn_threshold=warn_threshold, block_threshold=block_threshold, max_chars=max_chars,
                        forgery_action=Decision.WARN, warn_handling=warn_handling,
                        thresholds_source="EXPERIMENTAL context guard, ctx_dev; not validated")


def assess(text, guard, cfg: PolicyConfig | None = None, *, allow_guard_block: bool = False) -> PolicyResult:
    """allow_guard_block=False (default): the guard's score is clamped at the BLOCK threshold, so the guard
    alone can raise at most a held WARN (the policy BLOCKs only on p > block_threshold)."""
    cfg = cfg or config()
    is_text = isinstance(text, str)
    probability, error = None, None
    if is_text:
        try:
            probability = float(guard.probability(text))
            if not allow_guard_block:
                probability = min(probability, cfg.block_threshold)
        except Exception as e:  # fail closed, exactly like an Engine 2 failure
            error = f"context guard failed: {type(e).__name__}"
    forgery = find_forgery(text) if is_text else []
    res = evaluate(text, cfg, probability=probability, detector_error=error, forgery=forgery)
    notes = res.notes + ("score from the EXPERIMENTAL context guard (unvalidated); WARN is held for review",)
    return PolicyResult(res.decision, res.band, res.reasons, res.forward, res.held, res.flagged, notes)

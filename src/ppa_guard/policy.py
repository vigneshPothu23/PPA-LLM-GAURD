"""Risk assessment and security decision (fusion rules).

Order (most severe wins):
    1. Mandatory input rules set a floor (ALLOW / WARN / BLOCK).
    2. Engine 2 score gives a band:  score > block -> BLOCK ; score > warn -> WARN ; else ALLOW
       (strict ">" on the 2-decimal score, exactly like LLM Guard)
    3. final = max(floor, band). A low score can never lower a rule's decision.
    4. Detector failure -> BLOCK (fail closed).

Default thresholds (fixed before any test-set result was seen):
    block = 0.92  ProtectAI's own LLM Guard default
    warn  = 0.50  the model card's argmax: the model says INJECTION, but below ProtectAI's bar
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Iterable

from .engine1.official_constants import OFFICIAL_SEPARATORS
from .engine2.detector import MODEL_CARD_THRESHOLD, OFFICIAL_THRESHOLD, Detection


class Decision(IntEnum):
    ALLOW = 0
    WARN = 1
    BLOCK = 2


@dataclass(frozen=True)
class PolicyConfig:
    warn_threshold: float = MODEL_CARD_THRESHOLD
    block_threshold: float = OFFICIAL_THRESHOLD
    warn_may_proceed: bool = False             # WARN reaches Engine 1 only if explicitly allowed
    max_chars: int = 8000                       # longer input is not analysed -> BLOCK
    separator_forgery: Decision = Decision.WARN  # input contains a PPA boundary marker
    hidden_characters: Decision = Decision.WARN  # zero-width / bidi control characters
    truncated_analysis: Decision = Decision.WARN  # detector saw only the first 512 tokens

    def __post_init__(self) -> None:
        if not 0.0 <= self.warn_threshold <= self.block_threshold <= 1.0:
            raise ValueError("require 0 <= warn_threshold <= block_threshold <= 1")
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")


@dataclass(frozen=True)
class RuleHit:
    rule: str
    decision: Decision
    detail: str


@dataclass(frozen=True)
class Assessment:
    decision: Decision
    reasons: tuple[str, ...]
    rule_hits: tuple[RuleHit, ...]
    detector_band: Decision | None
    score: float | None
    forward_to_engine1: bool
    detection: Detection | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.name
        d["detector_band"] = self.detector_band.name if self.detector_band is not None else None
        d["rule_hits"] = [{"rule": h.rule, "decision": h.decision.name, "detail": h.detail} for h in self.rule_hits]
        return d


_ALLOWED_CF = {"­"}  # soft hyphen is common in pasted text


def hidden_characters(text: str) -> list[str]:
    return sorted({f"U+{ord(c):04X}" for c in text if unicodedata.category(c) == "Cf" and c not in _ALLOWED_CF})


def separator_markers(text: str, separators: Iterable[tuple[str, str]] = OFFICIAL_SEPARATORS) -> list[str]:
    hits: list[str] = []
    for left, right in separators:
        for m in (left, right):
            if m in text and m not in hits:
                hits.append(m)
    return hits


def apply_rules(text: str, cfg: PolicyConfig) -> list[RuleHit]:
    if not text.strip():
        return [RuleHit("empty_input", Decision.BLOCK, "input is empty")]
    hits: list[RuleHit] = []
    if len(text) > cfg.max_chars:
        hits.append(RuleHit("oversize_input", Decision.BLOCK,
                            f"{len(text)} characters exceeds the {cfg.max_chars}-character limit"))
    markers = separator_markers(text)
    if markers:
        hits.append(RuleHit("separator_forgery", cfg.separator_forgery,
                            f"input contains PPA boundary marker(s): {', '.join(markers[:3])}"))
    hidden = hidden_characters(text)
    if hidden:
        hits.append(RuleHit("hidden_characters", cfg.hidden_characters,
                            f"invisible/bidi control characters: {', '.join(hidden[:5])}"))
    return hits


def band(score: float, cfg: PolicyConfig) -> Decision:
    if score > cfg.block_threshold:
        return Decision.BLOCK
    if score > cfg.warn_threshold:
        return Decision.WARN
    return Decision.ALLOW


def assess(text: str, detection: Detection | None, cfg: PolicyConfig, detector_error: str | None = None) -> Assessment:
    hits = apply_rules(text, cfg)
    reasons = [f"[rule:{h.rule}] {h.detail} -> {h.decision.name}" for h in hits]
    floor = max((h.decision for h in hits), default=Decision.ALLOW)
    det_band = None
    score = None
    if detection is None:
        floor = Decision.BLOCK
        reasons.append(f"[engine2] unavailable ({detector_error or 'not run'}) -> BLOCK (fail closed)")
    else:
        score = detection.score
        det_band = band(score, cfg)
        reasons.append(f"[engine2] P(injection) = {score:.2f}; WARN if > {cfg.warn_threshold:.2f}, "
                       f"BLOCK if > {cfg.block_threshold:.2f} -> {det_band.name}")
        if detection.truncated:
            hits.append(RuleHit("truncated_analysis", cfg.truncated_analysis,
                                f"input is {detection.n_tokens} tokens; the model saw only the first 512"))
            reasons.append(f"[rule:truncated_analysis] {detection.n_tokens} tokens > 512 -> {cfg.truncated_analysis.name}")
            floor = max(floor, cfg.truncated_analysis)
    final = max(floor, det_band if det_band is not None else Decision.ALLOW)
    forward = final == Decision.ALLOW or (final == Decision.WARN and cfg.warn_may_proceed)
    return Assessment(final, tuple(reasons), tuple(hits), det_band, score, forward, detection, detector_error)

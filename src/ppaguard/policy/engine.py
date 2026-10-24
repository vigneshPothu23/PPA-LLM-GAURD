"""Central policy decision: turns raw signals into ALLOW / WARN / BLOCK.

Inputs are kept separate from the detector: Engine 2 contributes a calibrated probability, Engine 1 contributes
boundary-forgery findings, and the input itself contributes validation and hidden-character signals. Each signal maps to
a decision; the final decision is the most severe one. A failed or missing check is BLOCK (fail closed).

    detector band   P > block -> BLOCK,  P > warn -> WARN,  otherwise ALLOW     (strict ">")
    rules           invalid input -> BLOCK; detector error -> BLOCK;
                    strong boundary forgery / hidden control characters / partially analysed input -> configurable
    final           max(band, rules)

WARN handling is an application setting: "hold" (analyst review; nothing is forwarded) or "allow_with_flag".
A review release can only ever apply to a WARN (see `release_applies`); a BLOCK can never be released.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import IntEnum

from ..engine1.forgery import ForgeryFinding


class Decision(IntEnum):
    ALLOW = 0
    WARN = 1
    BLOCK = 2


@dataclass(frozen=True)
class PolicyConfig:
    warn_threshold: float
    block_threshold: float
    max_chars: int = 8000
    forgery_action: Decision = Decision.WARN
    hidden_chars_action: Decision = Decision.WARN
    partial_analysis_action: Decision = Decision.WARN
    warn_handling: str = "hold"                 # hold | allow_with_flag
    thresholds_source: str = "model card"

    def __post_init__(self):
        if not 0.0 < self.warn_threshold <= self.block_threshold < 1.0:
            raise ValueError("thresholds must satisfy 0 < warn <= block < 1")
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")
        if self.warn_handling not in ("hold", "allow_with_flag"):
            raise ValueError("warn_handling must be hold or allow_with_flag")
        for a in (self.forgery_action, self.hidden_chars_action, self.partial_analysis_action):
            if a not in (Decision.WARN, Decision.BLOCK):
                raise ValueError("rule actions must be WARN or BLOCK")


@dataclass(frozen=True)
class Reason:
    source: str           # input | engine2 | engine1 | rule | failsafe
    decision: Decision
    text: str

    def __str__(self) -> str:
        return f"[{self.source}] {self.text} -> {self.decision.name}"


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    band: Decision | None
    reasons: tuple[Reason, ...]
    forward: bool          # may Engine 1 assemble and the caller proceed?
    held: bool             # WARN held for analyst review
    flagged: bool          # WARN forwarded under allow_with_flag
    notes: tuple[str, ...] = field(default=())


_BIDI = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}


def hidden_characters(text: str) -> list[str]:
    """Invisible format characters (Unicode category Cf). Soft hyphen is ignored; a zero-width joiner between two
    symbols (emoji sequences) is ignored."""
    found = set()
    for i, ch in enumerate(text):
        cp = ord(ch)
        if unicodedata.category(ch) != "Cf" or cp == 0x00AD:
            continue
        if cp == 0x200D and 0 < i < len(text) - 1 and unicodedata.category(text[i - 1]) in ("So", "Sk", "Mn") \
                and unicodedata.category(text[i + 1]) in ("So", "Sk"):
            continue
        found.add(f"U+{cp:04X}{' (bidi)' if cp in _BIDI else ''}")
    return sorted(found)


def validate_input(text, cfg: PolicyConfig) -> str | None:
    if not isinstance(text, str):
        return "input is not text"
    if not text.strip():
        return "input is empty"
    if len(text) > cfg.max_chars:
        return f"input is {len(text)} characters; this application allows {cfg.max_chars}"
    if "\x00" in text:
        return "input contains NUL characters"
    return None


def evaluate(text, cfg: PolicyConfig, *, probability: float | None, detector_error: str | None,
             partial: bool = False, forgery: list[ForgeryFinding] | tuple = ()) -> PolicyResult:
    reasons: list[Reason] = []
    notes: list[str] = []
    bad = validate_input(text, cfg)
    if bad:
        reasons.append(Reason("input", Decision.BLOCK, bad))
    band = None
    if detector_error:
        reasons.append(Reason("failsafe", Decision.BLOCK, f"Engine 2 unavailable ({detector_error}); fail closed"))
    elif probability is None and not bad:
        reasons.append(Reason("failsafe", Decision.BLOCK, "no Engine 2 result; fail closed"))
    elif probability is not None:
        band = Decision.BLOCK if probability > cfg.block_threshold else Decision.WARN if probability > cfg.warn_threshold \
            else Decision.ALLOW
        reasons.append(Reason("engine2", band, f"P(injection) = {probability:.3f}; WARN if > {cfg.warn_threshold:.2f}, "
                                                f"BLOCK if > {cfg.block_threshold:.2f} ({cfg.thresholds_source})"))
    if isinstance(text, str) and not bad:
        hc = hidden_characters(text)
        if hc:
            reasons.append(Reason("rule", cfg.hidden_chars_action, "invisible control characters: " + ", ".join(hc[:6])))
        strong = [f for f in forgery if f.strong]
        if strong:
            reasons.append(Reason("engine1", cfg.forgery_action,
                                  "boundary forgery: " + "; ".join(f"{f.kind} '{f.excerpt}'" for f in strong[:3])))
        notes += [f"weak signal {f.kind}: '{f.excerpt}'" for f in forgery if not f.strong]
        if partial:
            reasons.append(Reason("rule", cfg.partial_analysis_action,
                                  "input longer than the detector's window budget; only the first part was analysed"))
    decision = max((r.decision for r in reasons), default=Decision.BLOCK)
    if not reasons:
        reasons.append(Reason("failsafe", Decision.BLOCK, "no checks could run; fail closed"))
    held = decision is Decision.WARN and cfg.warn_handling == "hold"
    flagged = decision is Decision.WARN and cfg.warn_handling == "allow_with_flag"
    forward = decision is Decision.ALLOW or flagged
    return PolicyResult(decision, band, tuple(reasons), forward, held, flagged, tuple(notes))


def release_applies(result: PolicyResult) -> bool:
    """An analyst release may only turn a held WARN into a forward. Never a BLOCK."""
    return result.decision is Decision.WARN and result.held

"""Industry policy profiles -> per-application policy, and the bounds within which an authorized user may edit them.

The detector defaults (WARN > 0.50 = model argmax, BLOCK > 0.92 = ProtectAI LLM Guard default) are the only
values with external provenance. Profiles that change them say so; the UI labels non-default thresholds as
"not validated".
"""

from __future__ import annotations

import json
import re

from ..policy import Decision, PolicyConfig

VALIDATED_WARN, VALIDATED_BLOCK = 0.50, 0.92

PROFILES = {
    "general": dict(
        label="General", warn_threshold=0.50, block_threshold=0.92, warn_action="hold", forgery_action="WARN",
        hidden_chars_action="WARN", truncated_action="WARN", max_chars=8000, retention_days=0, rate_limit_per_min=30,
        max_retention_days=30,
        note="Validated detector defaults. WARN is held for analyst review."),
    "software_it": dict(
        label="Software / IT", warn_threshold=0.50, block_threshold=0.92, warn_action="hold", forgery_action="WARN",
        hidden_chars_action="BLOCK", truncated_action="WARN", max_chars=12000, retention_days=7, rate_limit_per_min=60,
        max_retention_days=30,
        note="Invisible/bidi control characters are BLOCKED (Trojan-Source style code attacks). Longer inputs allowed "
             "for code; anything past 512 tokens is still flagged WARN because the detector does not see it."),
    "healthcare": dict(
        label="Healthcare", warn_threshold=0.40, block_threshold=0.85, warn_action="hold", forgery_action="BLOCK",
        hidden_chars_action="BLOCK", truncated_action="WARN", max_chars=8000, retention_days=0, rate_limit_per_min=30,
        max_retention_days=0,
        note="Stricter, NOT validated thresholds (0.40 / 0.85). No input text is ever retained."),
    "finance": dict(
        label="Finance", warn_threshold=0.50, block_threshold=0.90, warn_action="hold", forgery_action="BLOCK",
        hidden_chars_action="BLOCK", truncated_action="WARN", max_chars=8000, retention_days=0, rate_limit_per_min=30,
        max_retention_days=7,
        note="BLOCK threshold 0.90 is NOT validated. Boundary forgery is blocked outright."),
}

PURPOSE_TEMPLATES = {
    "coding": ("You are a code review assistant. Review the code submitted by the user. \n{user_input}\n",
               "REVIEW THE SUBMITTED CODE"),
    "email": ("Summarize the email provided by the user. \n{user_input}\n", "SUMMARIZE THE EMAIL"),
    "documents": ("Summarize the document provided by the user. \n{user_input}\n", "SUMMARIZE THE DOCUMENT"),
    "general": ("Answer the user's question helpfully and concisely. \n{user_input}\n", "ANSWER THE USER'S QUESTION"),
}

EDITABLE = ("warn_threshold", "block_threshold", "warn_action", "forgery_action", "hidden_chars_action",
            "truncated_action", "max_chars", "retention_days", "rate_limit_per_min")


class PolicyValidationError(ValueError):
    pass


def validate(profile: str, values: dict) -> dict:
    """Returns the cleaned editable fields or raises PolicyValidationError."""
    if profile not in PROFILES:
        raise PolicyValidationError(f"unknown profile {profile!r}")
    v = {k: values[k] for k in EDITABLE}
    w, b = float(v["warn_threshold"]), float(v["block_threshold"])
    if not (0.05 <= w <= b <= 0.99):
        raise PolicyValidationError("thresholds must satisfy 0.05 <= WARN <= BLOCK <= 0.99")
    for k in ("forgery_action", "hidden_chars_action", "truncated_action"):
        if v[k] not in ("WARN", "BLOCK"):
            raise PolicyValidationError(f"{k} must be WARN or BLOCK")
    if v["warn_action"] not in ("hold", "allow_with_flag"):
        raise PolicyValidationError("warn_action must be hold or allow_with_flag")
    if not 1 <= int(v["max_chars"]) <= 20000:
        raise PolicyValidationError("max_chars must be 1..20000")
    cap = PROFILES[profile]["max_retention_days"]
    if not 0 <= int(v["retention_days"]) <= cap:
        raise PolicyValidationError(f"retention_days must be 0..{cap} for the {PROFILES[profile]['label']} profile")
    if not 1 <= int(v["rate_limit_per_min"]) <= 600:
        raise PolicyValidationError("rate_limit_per_min must be 1..600")
    v.update(warn_threshold=round(w, 2), block_threshold=round(b, 2), max_chars=int(v["max_chars"]),
             retention_days=int(v["retention_days"]), rate_limit_per_min=int(v["rate_limit_per_min"]))
    return v


def defaults(profile: str) -> dict:
    return {k: PROFILES[profile][k] for k in EDITABLE}


def to_config(app: dict) -> PolicyConfig:
    """The engine-level policy for one application. WARN never auto-proceeds here; the gateway decides."""
    return PolicyConfig(warn_threshold=app["warn_threshold"], block_threshold=app["block_threshold"],
                        warn_may_proceed=False, max_chars=app["max_chars"],
                        separator_forgery=Decision[app["forgery_action"]],
                        hidden_characters=Decision[app["hidden_chars_action"]],
                        truncated_analysis=Decision[app["truncated_action"]])


def snapshot(app: dict) -> str:
    keys = ("profile",) + EDITABLE
    return json.dumps({k: app[k] for k in keys}, sort_keys=True)


def is_validated(app: dict) -> bool:
    return app["warn_threshold"] == VALIDATED_WARN and app["block_threshold"] == VALIDATED_BLOCK


# ---- data minimisation: redaction applied before any optional retention
_REDACT = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|xox[abp]|AKIA|ppag)[-_A-Za-z0-9]{12,}\b"), "[SECRET]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[NUMBER]"),
    (re.compile(r"\+?\d[\d ()-]{8,}\d"), "[PHONE]"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "[HEX]"),
]


def redact(text: str) -> str:
    for rx, rep in _REDACT:
        text = rx.sub(rep, text)
    return text

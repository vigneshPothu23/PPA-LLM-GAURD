"""Industry profiles and per-application policy settings.

Thresholds are model-specific, so an application normally uses the thresholds recommended in the active Engine 2
model card (selected on the frozen development set, or the provider's defaults for a third-party model). An authorised
user may override them per application; overrides are labelled "not validated" everywhere they are shown.
"""

from __future__ import annotations

import json
import re

from .engine import Decision, PolicyConfig

PROFILES = {
    "general": dict(label="General", forgery_action="WARN", hidden_chars_action="WARN", partial_analysis_action="WARN",
                    warn_handling="hold", max_chars=8000, retention_days=0, max_retention_days=30, rate_limit_per_min=30,
                    note="Model-recommended thresholds; suspicious boundaries and hidden characters are held for review."),
    "software_it": dict(label="Software / IT", forgery_action="WARN", hidden_chars_action="BLOCK",
                        partial_analysis_action="WARN", warn_handling="hold", max_chars=12000, retention_days=7,
                        max_retention_days=30, rate_limit_per_min=60,
                        note="Invisible/bidi control characters are BLOCKED (Trojan-Source style code attacks); longer "
                             "inputs allowed for code; redacted text kept 7 days for investigations."),
    "healthcare": dict(label="Healthcare", forgery_action="BLOCK", hidden_chars_action="BLOCK", partial_analysis_action="BLOCK",
                       warn_handling="hold", max_chars=8000, retention_days=0, max_retention_days=0, rate_limit_per_min=30,
                       note="Every rule signal BLOCKS; no input text is ever retained."),
    "finance": dict(label="Finance", forgery_action="BLOCK", hidden_chars_action="BLOCK", partial_analysis_action="WARN",
                    warn_handling="hold", max_chars=8000, retention_days=0, max_retention_days=7, rate_limit_per_min=30,
                    note="Boundary forgery and hidden characters BLOCK; retention at most 7 days."),
}

PURPOSE_TEMPLATES = {
    "coding": ("You are a code review assistant for the engineering team.", "review the submitted code and report problems"),
    "email": ("You are an e-mail assistant for company staff.", "summarise the e-mail and list any requested actions"),
    "documents": ("You are a documentation assistant.", "summarise the document accurately"),
    "general": ("You are a helpful assistant for company staff.", "answer the user's question helpfully and concisely"),
}

EDITABLE = ("warn_threshold", "block_threshold", "warn_handling", "forgery_action", "hidden_chars_action",
            "partial_analysis_action", "max_chars", "retention_days", "rate_limit_per_min")


class PolicyValidationError(ValueError):
    pass


def defaults(profile: str) -> dict:
    p = PROFILES[profile]
    return {"warn_threshold": None, "block_threshold": None, **{k: p[k] for k in EDITABLE if k in p}}


def validate(profile: str, values: dict) -> dict:
    if profile not in PROFILES:
        raise PolicyValidationError(f"unknown profile {profile!r}")
    v = {k: values.get(k) for k in EDITABLE}
    w, b = v["warn_threshold"], v["block_threshold"]
    if (w is None) != (b is None):
        raise PolicyValidationError("set both thresholds, or neither (use the model's recommended thresholds)")
    if w is not None:
        w, b = round(float(w), 2), round(float(b), 2)
        if not 0.05 <= w <= b <= 0.99:
            raise PolicyValidationError("thresholds must satisfy 0.05 <= WARN <= BLOCK <= 0.99")
    for k in ("forgery_action", "hidden_chars_action", "partial_analysis_action"):
        if v[k] not in ("WARN", "BLOCK"):
            raise PolicyValidationError(f"{k} must be WARN or BLOCK")
    if v["warn_handling"] not in ("hold", "allow_with_flag"):
        raise PolicyValidationError("warn_handling must be hold or allow_with_flag")
    try:
        mc, rd, rl = int(v["max_chars"]), int(v["retention_days"]), int(v["rate_limit_per_min"])
    except (TypeError, ValueError) as e:
        raise PolicyValidationError("max_chars, retention_days and rate_limit_per_min must be integers") from e
    if not 1 <= mc <= 20000:
        raise PolicyValidationError("max_chars must be 1..20000")
    cap = PROFILES[profile]["max_retention_days"]
    if not 0 <= rd <= cap:
        raise PolicyValidationError(f"retention_days must be 0..{cap} for the {PROFILES[profile]['label']} profile")
    if not 1 <= rl <= 600:
        raise PolicyValidationError("rate_limit_per_min must be 1..600")
    v.update(warn_threshold=w, block_threshold=b, max_chars=mc, retention_days=rd, rate_limit_per_min=rl)
    return v


def to_config(app: dict, recommended: dict) -> PolicyConfig:
    """`recommended` = the active model card's thresholds {"warn", "block", "context_warn", "source"}."""
    if app.get("warn_threshold") is not None:
        w, b, src = app["warn_threshold"], app["block_threshold"], "application override - not validated"
    else:
        w, b, src = recommended["warn"], recommended["block"], f"model card: {recommended.get('source', 'recommended')}"
    ctx_warn = recommended.get("context_warn", 0.95)  # default fallback
    return PolicyConfig(warn_threshold=w, block_threshold=b, max_chars=app["max_chars"],
                        forgery_action=Decision[app["forgery_action"]], hidden_chars_action=Decision[app["hidden_chars_action"]],
                        partial_analysis_action=Decision[app["partial_analysis_action"]], warn_handling=app["warn_handling"],
                        thresholds_source=src, context_warn_threshold=ctx_warn)


def snapshot(app: dict, cfg: PolicyConfig) -> str:
    return json.dumps({"profile": app["profile"], **{k: app[k] for k in EDITABLE},
                       "effective_warn": cfg.warn_threshold, "effective_block": cfg.block_threshold,
                       "thresholds_source": cfg.thresholds_source}, sort_keys=True)


def uses_override(app: dict) -> bool:
    return app.get("warn_threshold") is not None


# ---- data minimisation: redaction before any optional retention
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

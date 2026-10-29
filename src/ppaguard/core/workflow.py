"""The security workflow - the only path that produces a protected prompt.

  caller (signed-in user, or API key bound to one application)
   1  authenticate/authorize: role permission + organization + application scope   -> Forbidden / NotFound
   2  rate limit per caller per application                                         -> RateLimited (audited)
   3  input validation                                                              -> BLOCK
   4  Engine 2 threat analysis (ONNX transformer)                                   -> error => BLOCK (fail closed)
   5  Engine 1 boundary-forgery scan (signal only)
   6  policy decision ALLOW / WARN / BLOCK (ppaguard.policy)
   7  WARN: hold for analyst review, or forward with a flag (application setting);
      a held WARN may consume a single-use analyst release - only after this full re-check, and never for a BLOCK
   8  Engine 1 polymorphic assembly (only when forwarding)                          -> error => BLOCK
   9  one transaction: scan record + review + optional redacted text + audit event  -> failure => BLOCK, nothing returned
  10  result: what each engine found, the policy decision, and the action taken

Stored per request: SHA-256 of the input, its length, scores, decision metadata. Never the raw text (redacted text only
if the application enables retention) and never the separator of an assembled prompt.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Engine

from ..engine1 import Engine1Error, PolymorphicAssembler, find_forgery
from ..policy import Decision, profiles
from ..policy.engine import Reason, evaluate, release_applies, validate_input
from . import audit
from .authz import Forbidden, Principal, app_for, require
from .db import api_keys, new_id, now, retained_inputs, reviews, scans
from .keys import hash_key
from .roles import Role

NO_MODEL_THRESHOLDS = {"warn": 0.5, "block": 0.9, "source": "no model loaded"}


class RateLimited(Forbidden):
    pass


@dataclass
class WorkflowResult:
    decision: Decision
    outcome: str                         # protected_prompt_returned | held_for_review | blocked
    action: str                          # plain-language description of what happened
    scan_id: str | None
    reasons: list[str]
    notes: list[str] = field(default_factory=list)
    engine2: dict | None = None
    policy: dict | None = None
    engine1: dict | None = None
    review_id: str | None = None
    released: bool = False
    flagged: bool = False
    llm_note: str | None = None
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def protected_prompt(self) -> str | None:
        return self.engine1["prompt"] if self.engine1 and self.engine1.get("applied") else None

    @property
    def allowed(self) -> bool:
        return self.protected_prompt is not None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


class SecurityWorkflow:
    def __init__(self, engine: Engine, detector=None, detector_error: str | None = None, llm=None):
        self.engine = engine
        self.detector = detector
        self.detector_error = None if detector is not None else (detector_error or "Engine 2 model not loaded")
        self.llm = llm

    @property
    def recommended(self) -> dict:
        return dict(self.detector.card.thresholds) if self.detector is not None else NO_MODEL_THRESHOLDS

    # ------------------------------------------------------------------ entry points
    def submit_with_api_key(self, api_key: str, text: str, explain: bool = False) -> WorkflowResult:
        with self.engine.begin() as conn:
            k = conn.execute(select(api_keys).where(api_keys.c.key_hash == hash_key(api_key),
                                                    api_keys.c.revoked_at.is_(None))).mappings().first()
            if k is None:
                raise Forbidden("invalid or revoked API key")
            conn.execute(update(api_keys).where(api_keys.c.id == k["id"]).values(last_used_at=now()))
        p = Principal("api_key", k["org_id"], Role.MEMBER, app_id=k["app_id"], api_key_id=k["id"], label=k["prefix"])
        return self.submit(p, k["app_id"], text, explain=explain, channel="sdk")

    def submit(self, p: Principal, app_id: str, text, explain: bool = False, channel: str = "console") -> WorkflowResult:
        t0 = time.perf_counter()
        # 1-2 authorization, scope, rate limit
        with self.engine.begin() as conn:
            if p.kind == "user":
                require(p, "scan.run.any_app" if p.role in (Role.ADMIN, Role.ANALYST) else "scan.run.assigned_app")
            app = app_for(conn, p, app_id, "scan")
            who = scans.c.user_id == p.user_id if p.kind == "user" else scans.c.api_key_id == p.api_key_id
            recent = conn.execute(select(func.count()).select_from(scans).where(
                scans.c.org_id == p.org_id, scans.c.app_id == app_id, who,
                scans.c.created_at >= now() - timedelta(seconds=60))).scalar_one()
            if recent >= app["rate_limit_per_min"]:
                audit.append(conn, p.org_id, p.actor, "scan.rate_limited", f"app:{app_id}", {"limit": app["rate_limit_per_min"]})
                raise RateLimited(f"rate limit of {app['rate_limit_per_min']} scans per minute reached")
        cfg = profiles.to_config(app, self.recommended)
        is_text = isinstance(text, str)
        h = sha256(text) if is_text else sha256(repr(type(text)))

        # 3-4 input validation, Engine 2
        e2, det_err = None, None
        context_probability = None
        if validate_input(text, cfg) is None:
            if self.detector is None:
                det_err = self.detector_error
            else:
                try:
                    d = self.detector.detect(text, explain=explain)
                    e2 = {"probability": round(d.probability, 4), "raw_probability": round(d.raw_probability, 4),
                          "model_id": d.model_id, "provenance": d.provenance, "n_tokens": d.n_tokens, "windows": d.windows,
                          "partial": d.partial, "latency_ms": round(d.latency_ms, 1),
                          "evidence": [list(x) for x in d.evidence]}

                    # Engine 2 context signal (WARN only); an error here fails closed like any detector error
                    ctx = getattr(self.detector, "context_signal", None)
                    if ctx is not None:
                        c = ctx.score(text)
                        context_probability = round(c.probability, 4)
                        e2["context"] = {"probability": context_probability, "top_kind": c.top_kind,
                                         "n_segments": c.n_segments, "model_id": ctx.model_id,
                                         "top_segment": c.top_segment[:300]}
                except Exception as e:  # fail closed
                    det_err = f"{type(e).__name__}: {e}"
        if det_err:
            e2 = {"error": det_err, "model_id": getattr(self.detector, "model_id", None)}

        # 5-6 Engine 1 forgery signal, policy
        forgery = find_forgery(text) if is_text else []
        pr = evaluate(text, cfg, probability=e2.get("probability") if e2 and not det_err else None,
                      detector_error=det_err, partial=bool(e2 and e2.get("partial")), forgery=forgery,
                      context_probability=context_probability)
        reasons, notes = [str(r) for r in pr.reasons], list(pr.notes)
        decision, forward = pr.decision, pr.forward
        policy_info = {"decision": pr.decision.name, "detector_band": pr.band.name if pr.band is not None else None,
                       "warn_threshold": cfg.warn_threshold, "block_threshold": cfg.block_threshold,
                       "thresholds_source": cfg.thresholds_source, "warn_handling": cfg.warn_handling,
                       "held": pr.held, "flagged": pr.flagged, "profile": app["profile"]}

        # 8 Engine 1 assembly (prepared now, returned only after the record commits)
        e1 = {"applied": False, "forgery_findings": [asdict(f) | {"strong": f.strong} for f in forgery]}

        def assemble() -> bool:
            nonlocal decision
            try:
                a = PolymorphicAssembler(app["system_instructions"], app["task"], max_input_chars=cfg.max_chars).assemble(text)
            except Engine1Error as e:
                decision = Decision.BLOCK
                reasons.append(str(Reason("engine1", Decision.BLOCK, f"assembly refused: {e}")))
                return False
            e1.update(applied=True, prompt=a.prompt, template_id=a.template_id, separator_family=a.separator.family,
                      open=a.separator.open, close=a.separator.close, nonce=a.separator.nonce)
            return True

        if forward:
            forward = assemble()

        # 7 + 9 release consumption, record, audit - one transaction
        scan_id, review_id, released = new_id(), None, False
        try:
            with self.engine.begin() as conn:
                if release_applies(pr) and decision is Decision.WARN and p.kind == "user":
                    rel = conn.execute(select(reviews.c.id).where(
                        reviews.c.org_id == p.org_id, reviews.c.app_id == app_id, reviews.c.requester_id == p.user_id,
                        reviews.c.input_sha256 == h, reviews.c.status == "released",
                        reviews.c.release_expires_at > now()).order_by(reviews.c.decided_at)).first()
                    if rel is not None:
                        used = conn.execute(update(reviews).where(reviews.c.id == rel[0], reviews.c.status == "released")
                                            .values(status="release_used", release_used_at=now())).rowcount
                        if used == 1 and assemble():
                            forward, released = True, True
                            reasons.append(f"[review] single-use analyst release {rel[0][:8]} consumed after a full re-check -> forwarded")
                outcome = "protected_prompt_returned" if forward else ("held_for_review" if decision is Decision.WARN else "blocked")
                if outcome == "held_for_review":
                    review_id = new_id()
                conn.execute(insert(scans).values(
                    id=scan_id, org_id=p.org_id, app_id=app_id, user_id=p.user_id, api_key_id=p.api_key_id, channel=channel,
                    created_at=now(), input_sha256=h, input_chars=len(text) if is_text else 0,
                    n_tokens=e2.get("n_tokens") if e2 else None, partial=e2.get("partial") if e2 else None,
                    score=e2.get("probability") if e2 else None, detector_band=policy_info["detector_band"],
                    decision=decision.name, outcome=outcome, reasons_json=json.dumps(reasons),
                    policy_json=profiles.snapshot(app, cfg), model_id=getattr(self.detector, "model_id", None),
                    engine1_applied=bool(forward), engine1_template=e1.get("template_id") if forward else None,
                    review_id=review_id, latency_ms=(time.perf_counter() - t0) * 1000))
                if review_id:
                    conn.execute(insert(reviews).values(id=review_id, org_id=p.org_id, app_id=app_id, scan_id=scan_id,
                                                        requester_id=p.user_id, input_sha256=h, status="open", created_at=now()))
                if is_text and app["retention_days"] > 0 and text.strip():
                    conn.execute(insert(retained_inputs).values(scan_id=scan_id, org_id=p.org_id, redacted_text=profiles.redact(text),
                                                                expires_at=now() + timedelta(days=app["retention_days"])))
                audit.append(conn, p.org_id, p.actor, "scan.decision", f"scan:{scan_id}",
                             {"app_id": app_id, "decision": decision.name, "outcome": outcome,
                              "score": e2.get("probability") if e2 else None, "released": released, "flagged": pr.flagged,
                              "channel": channel})
        except Exception as e:  # fail closed: no record, no protected prompt
            return WorkflowResult(Decision.BLOCK, "blocked", "Blocked: the decision could not be recorded, so nothing was released.",
                                  None, reasons + [f"[failsafe] record could not be written ({type(e).__name__}) -> BLOCK"],
                                  notes, e2, policy_info, {"applied": False}, latency_ms=(time.perf_counter() - t0) * 1000,
                                  errors=[repr(e)])

        if not forward:
            e1 = {k: v for k, v in e1.items() if k in ("applied", "forgery_findings")} | {"applied": False}
        action = {"protected_prompt_returned": "Forwarded: Engine 1 wrapped the input in a fresh per-request boundary; the "
                                               "protected prompt can be sent to the application's LLM."
                                               + (" (analyst release used)" if released else "")
                                               + (" Flagged under the application's WARN setting." if pr.flagged else ""),
                  "held_for_review": "Held for a Security Analyst. Nothing was forwarded.",
                  "blocked": "Blocked. Nothing was forwarded."}[outcome]
        llm_note = None
        if forward:
            if self.llm is None:
                llm_note = "No LLM is connected in PPA Guard; the calling application sends the protected prompt to its own LLM."
            else:
                llm_note = self.llm(e1["prompt"])
        return WorkflowResult(decision, outcome, action, scan_id, reasons, notes, e2, policy_info, e1, review_id, released,
                              pr.flagged, llm_note, (time.perf_counter() - t0) * 1000)

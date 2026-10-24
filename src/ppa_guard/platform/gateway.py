"""The enforced request path. Everything that produces a protected prompt goes through Gateway.submit().

  identify + authorize (Rule 1, 2) -> application scope (Rule 4) -> rate limit -> Engine 2 detection
  -> policy decision (ALLOW / WARN / BLOCK) -> WARN handling (hold for review, or a single-use analyst release)
  -> Engine 1 PPA assembly (only when forwarding) -> durable record + audit event (one transaction)
  -> protected prompt returned to the caller. Nothing is returned if the record cannot be written.

Fail closed: detector missing or erroring -> BLOCK; record failure -> BLOCK; any authorization doubt -> refused.
Data minimisation (Rule 5): only a SHA-256 of the input, its length and the decision metadata are stored.
Redacted text is kept only if the application enables retention, and only until it expires. Evidence sentences
are returned to the caller but never stored.

Supported integration boundary: this is a Python call. The Streamlit console calls it, and an approved Python
application can call it in-process with an application API key (submit_with_api_key). It is NOT a network proxy:
it cannot intercept traffic from applications that do not call it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Engine

from ..engine1 import PolymorphicPromptAssembler, SeparatorCollisionError
from ..llm import NotConnectedLLM
from ..policy import Decision, assess
from . import audit, policies
from .authz import Forbidden, Principal, app_for, require
from .db import api_keys, new_id, now, retained_inputs, reviews, scans
from .roles import Role


class RateLimited(Forbidden):
    pass


@dataclass
class GuardResult:
    decision: Decision
    outcome: str                         # protected_prompt_returned | held_for_review | blocked
    scan_id: str | None
    reasons: tuple[str, ...]
    score: float | None = None
    n_tokens: int | None = None
    truncated: bool | None = None
    evidence: tuple = ()
    protected_prompt: str | None = None
    separator: tuple[str, str] | None = None
    review_id: str | None = None
    released: bool = False
    flagged: bool = False
    model_id: str | None = None
    policy_validated: bool = True
    llm_note: str | None = None
    latency_ms: float = 0.0
    errors: list = field(default_factory=list)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Gateway:
    def __init__(self, engine: Engine, detector=None, detector_error: str | None = None, llm=None):
        self.engine = engine
        self.detector = detector
        self.detector_error = detector_error if detector is None else None
        self.llm = llm or NotConnectedLLM()

    # ------------------------------------------------------------------ entry points
    def submit_with_api_key(self, api_key: str, text: str, explain: bool = False) -> GuardResult:
        with self.engine.begin() as conn:
            k = conn.execute(select(api_keys).where(api_keys.c.key_hash == hash_key(api_key or ""),
                                                    api_keys.c.revoked_at.is_(None))).mappings().first()
            if k is None:
                raise Forbidden("invalid or revoked API key")
            conn.execute(update(api_keys).where(api_keys.c.id == k["id"]).values(last_used_at=now()))
        p = Principal("api_key", k["org_id"], Role.MEMBER, app_id=k["app_id"], api_key_id=k["id"], label=k["prefix"])
        return self.submit(p, k["app_id"], text, explain=explain, channel="sdk")

    def submit(self, p: Principal, app_id: str, text: str, explain: bool = False, channel: str = "console") -> GuardResult:
        t0 = time.perf_counter()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # 1-2. authorization + application scope + rate limit (Rules 2, 3, 4)
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
        cfg = policies.to_config(app)
        h = sha256(text)

        # 3. Engine 2 (outside the transaction: it is the slow step)
        detection, err = None, None
        if text.strip() and len(text) <= cfg.max_chars:
            if self.detector is None:
                err = self.detector_error or "Engine 2 not loaded"
            else:
                try:
                    detection = self.detector.detect(text, explain=explain)
                except Exception as e:  # fail closed
                    err = f"{type(e).__name__}: {e}"
        else:
            err = "input rejected before detection"

        # 4. policy decision (existing engine-level logic, unchanged)
        a = assess(text, detection, cfg, err)
        decision, reasons = a.decision, list(a.reasons)
        forward, flagged = decision is Decision.ALLOW, False
        if decision is Decision.WARN and app["warn_action"] == "allow_with_flag":
            forward, flagged = True, True
            reasons.append("[policy] WARN forwarded with a flag (application setting)")

        # 5. Engine 1, prepared now, returned only after the record commits
        protected, sep = None, None
        released, review_id = False, None

        def assemble():
            nonlocal decision, protected, sep
            try:
                asm = PolymorphicPromptAssembler(app["system_prompt"], app["task_topic"], hardened=True).single_prompt_assemble(text)
                protected, sep = asm.prompt, asm.canary
                return True
            except SeparatorCollisionError as e:
                decision = Decision.BLOCK
                reasons.append(f"[engine1] {e} -> BLOCK")
                return False

        if forward:
            forward = assemble()

        # 6. record + WARN handling + audit, one transaction (nothing leaves without a record)
        scan_id = new_id()
        try:
            with self.engine.begin() as conn:
                if decision is Decision.WARN and not forward and p.kind == "user":
                    rel = conn.execute(select(reviews.c.id).where(
                        reviews.c.org_id == p.org_id, reviews.c.app_id == app_id, reviews.c.requester_id == p.user_id,
                        reviews.c.input_sha256 == h, reviews.c.status == "released",
                        reviews.c.release_expires_at > now()).order_by(reviews.c.decided_at)).first()
                    if rel is not None:
                        used = conn.execute(update(reviews).where(reviews.c.id == rel[0], reviews.c.status == "released")
                                            .values(status="release_used", release_used_at=now())).rowcount
                        if used == 1 and assemble():
                            forward, released = True, True
                            reasons.append(f"[review] single-use analyst release {rel[0][:8]} consumed after a full re-scan")
                outcome = "protected_prompt_returned" if forward else ("held_for_review" if decision is Decision.WARN else "blocked")
                if outcome == "held_for_review":
                    review_id = new_id()
                det = a.detection
                conn.execute(insert(scans).values(
                    id=scan_id, org_id=p.org_id, app_id=app_id, user_id=p.user_id, api_key_id=p.api_key_id, channel=channel,
                    created_at=now(), input_sha256=h, input_chars=len(text),
                    n_tokens=det.n_tokens if det else None, truncated=det.truncated if det else None,
                    score=a.score, detector_band=a.detector_band.name if a.detector_band is not None else None,
                    decision=decision.name, outcome=outcome, reasons_json=json.dumps(reasons), policy_json=policies.snapshot(app),
                    model_id=getattr(self.detector, "model_id", None), engine1_applied=bool(forward), review_id=review_id,
                    latency_ms=(time.perf_counter() - t0) * 1000))
                if review_id:
                    conn.execute(insert(reviews).values(id=review_id, org_id=p.org_id, app_id=app_id, scan_id=scan_id,
                                                        requester_id=p.user_id, input_sha256=h, status="open", created_at=now()))
                if app["retention_days"] > 0 and text.strip():
                    conn.execute(insert(retained_inputs).values(scan_id=scan_id, org_id=p.org_id, redacted_text=policies.redact(text),
                                                                expires_at=now() + timedelta(days=app["retention_days"])))
                audit.append(conn, p.org_id, p.actor, "scan.decision", f"scan:{scan_id}",
                             {"app_id": app_id, "decision": decision.name, "outcome": outcome, "score": a.score,
                              "released": released, "flagged": flagged})
        except Exception as e:  # fail closed: no record, no protected prompt
            return GuardResult(Decision.BLOCK, "blocked", None, tuple(reasons) + (f"[record] could not be written ({type(e).__name__}) -> BLOCK",),
                               score=a.score, model_id=getattr(self.detector, "model_id", None),
                               latency_ms=(time.perf_counter() - t0) * 1000, errors=[repr(e)])

        note = None
        if forward:
            note = self.llm.generate(protected).text  # placeholder adapter: no LLM is contacted
        return GuardResult(decision, outcome, scan_id, tuple(reasons), score=a.score,
                           n_tokens=a.detection.n_tokens if a.detection else None,
                           truncated=a.detection.truncated if a.detection else None,
                           evidence=a.detection.evidence if a.detection else (),
                           protected_prompt=protected if forward else None, separator=sep if forward else None,
                           review_id=review_id, released=released, flagged=flagged,
                           model_id=getattr(self.detector, "model_id", None), policy_validated=policies.is_validated(app),
                           llm_note=note, latency_ms=(time.perf_counter() - t0) * 1000)

"""Python integration interface.

An approved Python application screens untrusted text through the full PPA Guard workflow with an application API key
(created by the application's owner or a Platform Administrator in the console). The key is bound to one application,
can be revoked, and is the only credential the client accepts.

    from ppaguard.sdk import PPAGuardClient

    guard = PPAGuardClient(api_key=os.environ["PPAGUARD_API_KEY"])          # same database as the console
    r = guard.protect(untrusted_text)
    if r.allowed:
        reply = my_llm(r.protected_prompt)          # your own LLM call; PPA Guard never calls one
        guard.check_response(reply, r)              # optional: did the model echo the boundary marker?
    else:
        show(r.decision, r.reasons)                 # BLOCK, or WARN held for review (resubmit after a release)

Boundary: this is an in-process library sharing the console's database. It protects the calls an application makes
through it; it cannot see or stop traffic from code that does not call it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .core.db import make_engine
from .core.runtime import load_detector
from .core.workflow import SecurityWorkflow, WorkflowResult
from .engine1 import PolymorphicAssembler
from .engine1.assembler import AssembledPrompt
from .engine1.separators import SeparatorPair


@dataclass(frozen=True)
class GuardResponse:
    decision: str                   # ALLOW | WARN | BLOCK
    allowed: bool                   # True only when a protected prompt was produced
    action: str
    protected_prompt: str | None
    reasons: tuple[str, ...]
    scan_id: str | None
    review_id: str | None
    probability: float | None
    model_id: str | None
    _separator: SeparatorPair | None = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.allowed


class PPAGuardClient:
    def __init__(self, api_key: str, database_url: str | None = None, detector=None, environ=None):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("an application API key is required")
        self._key = api_key
        env = os.environ if environ is None else environ
        self._engine = make_engine(database_url or env.get("PPAGUARD_DATABASE_URL"))
        if detector is None:
            detector, status = load_detector(env)
            err = status.error
        else:
            err = None
        self.model_id = getattr(detector, "model_id", None)
        self._wf = SecurityWorkflow(self._engine, detector, err)

    def protect(self, text: str, explain: bool = False) -> GuardResponse:
        """Runs the full workflow. Raises ppaguard.core.authz.Forbidden for an invalid/revoked key or a paused app."""
        r: WorkflowResult = self._wf.submit_with_api_key(self._key, text, explain=explain)
        sep = SeparatorPair(r.engine1["separator_family"], "", r.engine1["nonce"], r.engine1["open"], r.engine1["close"]) \
            if r.allowed else None
        return GuardResponse(r.decision.name, r.allowed, r.action, r.protected_prompt, tuple(r.reasons), r.scan_id,
                             r.review_id, (r.engine2 or {}).get("probability"), (r.engine2 or {}).get("model_id"), sep)

    @staticmethod
    def check_response(llm_response: str, response: GuardResponse) -> bool:
        """True if the LLM output reproduced this request's boundary marker (a sign the model echoed the prompt
        structure). Only meaningful for an allowed response."""
        if not response._separator or not response.protected_prompt:
            return False
        fake = AssembledPrompt(response.protected_prompt, response._separator, "", "", 0, 0)
        return PolymorphicAssembler.check_response(llm_response, fake).marker_leaked


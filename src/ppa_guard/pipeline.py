"""Fusion pipeline:  input -> Engine 2 (DeBERTa) -> policy -> [Engine 1 (PPA) -> LLM placeholder]

Engine 1 is called ONLY when the assessment permits it; the LLM placeholder ONLY after Engine 1.
Nothing in this module calls a real LLM.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

from .engine1 import AssembledPrompt, PolymorphicPromptAssembler, SeparatorCollisionError
from .engine2.detector import Detection
from .llm import LLMClient, LLMResponse, NotConnectedLLM
from .policy import Assessment, Decision, PolicyConfig, assess


class Detector(Protocol):
    model_id: str

    def detect(self, text: str, explain: bool = False) -> Detection: ...


@dataclass(frozen=True)
class PipelineResult:
    assessment: Assessment
    protected: AssembledPrompt | None
    llm_response: LLMResponse | None
    engine1_invoked: bool
    input_sha256: str
    latency_ms: float

    @property
    def decision(self) -> Decision:
        return self.assessment.decision


class SecurityPipeline:
    def __init__(self, detector: Detector, assembler: PolymorphicPromptAssembler, policy: PolicyConfig,
                 llm: LLMClient | None = None):
        self.detector, self.assembler, self.policy = detector, assembler, policy
        self.llm = llm or NotConnectedLLM()

    def process(self, text: str, explain: bool = False) -> PipelineResult:
        if not isinstance(text, str):
            raise TypeError("input must be a str")
        t0 = time.perf_counter()
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        detection, error = None, None
        if text.strip() and len(text) <= self.policy.max_chars:
            try:
                detection = self.detector.detect(text, explain=explain)
            except Exception as exc:  # fail closed; never skip detection silently
                error = f"{type(exc).__name__}: {exc}"
        else:
            error = "input rejected before detection"
        a = assess(text, detection, self.policy, error)

        protected, response, invoked = None, None, False
        if a.forward_to_engine1:
            invoked = True
            try:
                protected = self.assembler.single_prompt_assemble(text)
                response = self.llm.generate(protected.prompt)
            except SeparatorCollisionError as exc:
                a = Assessment(Decision.BLOCK, a.reasons + (f"[engine1] {exc} -> BLOCK",), a.rule_hits,
                               a.detector_band, a.score, False, a.detection, a.error)
        return PipelineResult(a, protected, response, invoked, digest, (time.perf_counter() - t0) * 1000)

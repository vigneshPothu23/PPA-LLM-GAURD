"""LLM interface -- placeholder only (no LLM is called in this project's scope).

The pipeline hands an Engine-1-protected prompt to an `LLMClient`. The default
`NotConnectedLLM` never contacts any service; it returns a fixed notice. A future
integration implements `generate()` and, if it wants, uses PPA's canary check
(`PolymorphicPromptAssembler.leak_detect`) on the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    connected: bool
    provider: str
    separator_leaked: bool | None = None  # filled when a real response is checked with PPA's leak_detect


class LLMClient(Protocol):
    provider: str

    def generate(self, prompt: str) -> LLMResponse: ...


class NotConnectedLLM:
    provider = "none (placeholder)"

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="[No LLM connected. In a full deployment, the protected prompt above would be sent to the agent's LLM.]",
            connected=False,
            provider=self.provider,
        )

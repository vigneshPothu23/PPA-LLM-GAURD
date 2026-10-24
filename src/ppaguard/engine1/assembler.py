"""Engine 1 - polymorphic prompt assembly.

For every request: draw a fresh separator pair (random family, tag and nonce), pick an instruction template variant,
and place the untrusted text between the separators inside the trusted instructions. The input text itself is never
modified. Assembly refuses (raises) instead of producing a prompt whose boundaries could be ambiguous.
"""

from __future__ import annotations

import hashlib
import random
import secrets
from dataclasses import dataclass

from .separators import SeparatorGenerator, SeparatorPair
from .templates import TEMPLATES, Template


class Engine1Error(Exception):
    """Base class: the workflow treats any Engine 1 error as a reason to BLOCK."""


class InvalidInput(Engine1Error):
    pass


class SeparatorCollision(Engine1Error):
    pass


@dataclass(frozen=True)
class AssembledPrompt:
    prompt: str
    separator: SeparatorPair
    template_id: str
    input_sha256: str
    data_start: int          # character offsets of the untrusted text inside `prompt`
    data_end: int

    @property
    def data(self) -> str:
        return self.prompt[self.data_start:self.data_end]


@dataclass(frozen=True)
class ResponseCheck:
    marker_leaked: bool
    details: tuple[str, ...]


class PolymorphicAssembler:
    def __init__(self, system_instructions: str, task: str, *, max_input_chars: int = 20000,
                 rng: random.Random | None = None, templates: tuple[Template, ...] = TEMPLATES,
                 generator: SeparatorGenerator | None = None, max_draws: int = 16):
        if not isinstance(system_instructions, str) or not system_instructions.strip():
            raise ValueError("system_instructions must be a non-empty string")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if not templates:
            raise ValueError("at least one template is required")
        for t in templates:
            for ph in ("{system}", "{task}", "{open}", "{close}", "{data}"):
                if ph not in t.text:
                    raise ValueError(f"template {t.id} lacks placeholder {ph}")
            if t.text.count("{data}") != 1:
                raise ValueError(f"template {t.id} must contain {{data}} exactly once")
        self.system, self.task = system_instructions.strip(), task.strip()
        self.max_input_chars, self.max_draws = max_input_chars, max_draws
        self.rng = rng or secrets.SystemRandom()
        self.templates = templates
        self.generator = generator or SeparatorGenerator(rng=self.rng)

    def assemble(self, untrusted_text: str) -> AssembledPrompt:
        if not isinstance(untrusted_text, str):
            raise InvalidInput("input must be a string")
        if not untrusted_text.strip():
            raise InvalidInput("input is empty")
        if len(untrusted_text) > self.max_input_chars:
            raise InvalidInput(f"input longer than {self.max_input_chars} characters")
        if "\x00" in untrusted_text:
            raise InvalidInput("input contains NUL characters")
        trusted = self.system + "\n" + self.task
        for _ in range(self.max_draws):
            sep = self.generator.draw()
            if not sep.occurs_in(untrusted_text) and not sep.occurs_in(trusted):
                break
        else:
            raise SeparatorCollision(f"no collision-free separator after {self.max_draws} draws")
        tpl = self.rng.choice(self.templates)
        # Fill the trusted parts first, then insert the data at its single slot, so braces inside the untrusted text
        # are never interpreted as format fields.
        head, tail = tpl.text.split("{data}")
        fill = dict(system=self.system, task=self.task, open=sep.open, close=sep.close)
        head, tail = head.format(**fill), tail.format(**fill)
        prompt = head + untrusted_text + tail
        return AssembledPrompt(prompt, sep, tpl.id, hashlib.sha256(untrusted_text.encode("utf-8", "surrogatepass")).hexdigest(),
                               len(head), len(head) + len(untrusted_text))

    @staticmethod
    def check_response(response: str, assembled: AssembledPrompt) -> ResponseCheck:
        """For use when an LLM is connected: a response that reproduces the per-request marker or nonce shows that the
        model echoed the prompt structure (a sign of prompt leakage or a successful injection)."""
        found = []
        low = (response or "").lower()
        for label, s in (("opening marker", assembled.separator.open), ("closing marker", assembled.separator.close),
                         ("nonce", assembled.separator.nonce)):
            if s.lower() in low:
                found.append(label)
        return ResponseCheck(bool(found), tuple(found))

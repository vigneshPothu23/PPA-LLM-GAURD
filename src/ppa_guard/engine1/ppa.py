"""Engine 1 -- Polymorphic Prompt Assembling (PPA).

Reimplementation of:
    Z. Wang, N. Nagaraja, L. Zhang, H. Bahsi, P. Patil, P. Liu,
    "To Protect the LLM Agent Against the Prompt Injection Attack with
    Polymorphic Prompt", DSN-S 2025, pp. 22-28, doi:10.1109/DSN-S65789.2025.00037
    (arXiv:2506.05739).

and of the authors' reference SDK (MIT licensed):
    https://github.com/zhilongwang/LLMAgentProtector @ b3c20a3

Engine 1 is prompt-structure protection only. It does not classify input,
it does not call an LLM, and it has no ML dependencies.

Two modes
---------
faithful (default):
    Byte-for-byte the same assembled prompt as the reference SDK for the
    same separator choice (verified by tests/test_engine1_equivalence.py).
    The only difference is the random source: a CSPRNG
    (``secrets.SystemRandom``) instead of Python's Mersenne Twister. The
    sampling distribution (uniform choice over the list) is unchanged.

hardened (our extension, NOT part of the paper):
    Same assembly, but a separator whose begin/end string already occurs in
    the user input is never chosen. If every separator collides, assembly
    is refused with ``SeparatorCollisionError``.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass, field
from typing import Sequence

from .official_constants import (
    ANTI_PROMPT_LEAKAGE_CONSTRAIN,
    FORMAT_CONSTRAIN,
    OFFICIAL_SEPARATORS,
    TOPIC_CONSTRAIN,
)

Separator = tuple[str, str]

# Reference SDK demo task (kept verbatim, including its wording).
DEFAULT_SYSTEM_PROMPT = "Please summary the following article from user. \n{user_input}\n"
DEFAULT_TASK_TOPIC = "!!!SUMMARY THE ARTICLE FROM USER"


class SeparatorCollisionError(ValueError):
    """Raised in hardened mode when every separator occurs in the input."""


@dataclass(frozen=True)
class AssembledPrompt:
    mode: str                      # "single" or "double"
    canary: Separator              # (left_sep, right_sep) actually used
    prompt: str | None = None      # single-prompt mode
    system_prompt: str | None = None  # double-prompt mode
    user_prompt: str | None = None    # double-prompt mode
    colliding_separators: tuple[Separator, ...] = field(default_factory=tuple)
    hardened: bool = False


def find_collisions(user_input: str, separators: Sequence[Separator]) -> tuple[Separator, ...]:
    """Return the separator pairs whose begin or end marker appears in the input."""
    seen: list[Separator] = []
    for left, right in separators:
        if (left in user_input or right in user_input) and (left, right) not in seen:
            seen.append((left, right))
    return tuple(seen)


class PolymorphicPromptAssembler:
    """PPA assembler (paper Algorithm 1 / reference SDK class of the same name)."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        task_topic: str = DEFAULT_TASK_TOPIC,
        separators: Sequence[Separator] = OFFICIAL_SEPARATORS,
        rng: random.Random | None = None,
        hardened: bool = False,
    ) -> None:
        if not separators:
            raise ValueError("separator list must not be empty")
        for pair in separators:
            if len(pair) != 2 or not all(isinstance(s, str) and s for s in pair):
                raise ValueError(f"invalid separator pair: {pair!r}")
        self.separators: tuple[Separator, ...] = tuple((a, b) for a, b in separators)
        # CSPRNG by default; pass random.Random(seed) for reproducible experiments.
        self._rng = rng if rng is not None else secrets.SystemRandom()
        self.hardened = hardened
        # Identical construction to the reference SDK __init__.
        self.secure_system_prompt = (
            system_prompt
            + ANTI_PROMPT_LEAKAGE_CONSTRAIN
            + "\n\n"
            + TOPIC_CONSTRAIN.replace("{task_topic}", task_topic)
        )

    # -- separator choice -------------------------------------------------
    def _choose(self, user_input: str) -> tuple[Separator, tuple[Separator, ...]]:
        collisions = find_collisions(user_input, self.separators)
        if not self.hardened:
            return self._rng.choice(self.separators), collisions
        colliding = set(collisions)
        pool = [s for s in self.separators if s not in colliding]
        if not pool:
            raise SeparatorCollisionError(
                "every available separator occurs in the user input; refusing to assemble"
            )
        return self._rng.choice(pool), collisions

    @staticmethod
    def _wrap(left: str, right: str, user_input: str) -> tuple[str, str]:
        return FORMAT_CONSTRAIN.format(left_sep=left, right_sep=right), (
            "\n\n" + left + "\n" + user_input + "\n" + right + "\n"
        )

    # -- public API (mirrors the reference SDK) ---------------------------
    def single_prompt_assemble(self, user_input: str) -> AssembledPrompt:
        if not isinstance(user_input, str):
            raise TypeError("user_input must be a str")
        (left, right), collisions = self._choose(user_input)
        constraint, wrapped = self._wrap(left, right, user_input)
        block = "\n" + constraint + wrapped
        prompt = self.secure_system_prompt.replace("{user_input}", block)
        return AssembledPrompt(
            mode="single",
            canary=(left, right),
            prompt=prompt,
            colliding_separators=collisions,
            hardened=self.hardened,
        )

    def double_prompt_assemble(self, user_input: str) -> AssembledPrompt:
        """System/user split. Faithful to the SDK, which leaves the system prompt
        unmodified (any literal ``{user_input}`` placeholder stays in it)."""
        if not isinstance(user_input, str):
            raise TypeError("user_input must be a str")
        (left, right), collisions = self._choose(user_input)
        constraint, wrapped = self._wrap(left, right, user_input)
        return AssembledPrompt(
            mode="double",
            canary=(left, right),
            system_prompt=self.secure_system_prompt,
            user_prompt=constraint + wrapped,
            colliding_separators=collisions,
            hardened=self.hardened,
        )

    @staticmethod
    def leak_detect(response: str, canary: Separator) -> bool:
        """Reference SDK v0.1.0: a response that echoes a separator indicates leakage."""
        left, right = canary
        return left in response or right in response


# -- Paper Section IV, equations (1)-(3) ----------------------------------
def breach_probability(n: int, p_i: float) -> float:
    """Eq. (1): P = 1/n + (n-1)/n * P_i  (attacker guesses separator S_i)."""
    _check(n, [p_i])
    return 1.0 / n + (n - 1) / n * p_i


def whitebox_breach_probability(p: Sequence[float]) -> float:
    """Eq. (2): P_w = 1/n + (n-1)/n * mean(P_i)  (attacker knows the list)."""
    n = len(p)
    _check(n, p)
    return 1.0 / n + (n - 1) / n * (sum(p) / n)


def blackbox_breach_probability(p: Sequence[float]) -> float:
    """Eq. (3): P_b = (n-1)/n * mean(P_i)  (attacker does not know the list)."""
    n = len(p)
    _check(n, p)
    return (n - 1) / n * (sum(p) / n)


def _check(n: int, p: Sequence[float]) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if any(not 0.0 <= x <= 1.0 for x in p):
        raise ValueError("probabilities must be within [0, 1]")

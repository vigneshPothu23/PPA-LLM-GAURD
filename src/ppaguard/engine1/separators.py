"""Separator generation for polymorphic prompt assembly.

Every request gets a fresh pair of boundary markers drawn from several marker families, each carrying a random nonce
(48 bits by default). An attacker who does not see the assembled prompt cannot know the closing marker in advance, so
text that "closes" the data block and starts new instructions cannot match it.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass

# (family name, opening format, closing format). {tag} is a word, {nonce} the random hex string.
FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("square", "[[{tag}-{nonce}]]", "[[/{tag}-{nonce}]]"),
    ("angle", "<{tag}_{nonce}>", "</{tag}_{nonce}>"),
    ("hash", "##### {tag} {nonce} BEGIN #####", "##### {tag} {nonce} END #####"),
    ("at", "@@@ BEGIN {tag} {nonce} @@@", "@@@ END {tag} {nonce} @@@"),
    ("brace", "{{{{{tag}:{nonce}}}}}", "{{{{/{tag}:{nonce}}}}}"),
    ("equals", "===[{nonce} {tag} START]===", "===[{nonce} {tag} STOP]==="),
    ("pipe", "|~{tag}~{nonce}~|", "|~/{tag}~{nonce}~|"),
    ("percent", "%%% {nonce}:{tag}:open %%%", "%%% {nonce}:{tag}:close %%%"),
)
TAGS: tuple[str, ...] = ("DATA", "INPUT", "CONTENT", "UNTRUSTED", "PAYLOAD", "USER_TEXT", "DOCUMENT", "QUOTED")


@dataclass(frozen=True)
class SeparatorPair:
    family: str
    tag: str
    nonce: str
    open: str
    close: str

    def occurs_in(self, text: str) -> bool:
        low = text.lower()
        return self.open.lower() in low or self.close.lower() in low or self.nonce.lower() in low


class SeparatorGenerator:
    """Draws separator pairs. `rng` defaults to a cryptographically secure source; tests may pass a seeded Random."""

    def __init__(self, nonce_bytes: int = 6, rng: random.Random | None = None,
                 families: tuple[tuple[str, str, str], ...] = FAMILIES, tags: tuple[str, ...] = TAGS):
        if nonce_bytes < 4:
            raise ValueError("nonce_bytes must be at least 4 (32 bits)")
        if not families or not tags:
            raise ValueError("families and tags must not be empty")
        self.nonce_bytes = nonce_bytes
        self.rng = rng or secrets.SystemRandom()
        self.families = families
        self.tags = tags

    def _nonce(self) -> str:
        if isinstance(self.rng, secrets.SystemRandom):
            return secrets.token_hex(self.nonce_bytes).upper()
        return "".join(self.rng.choice("0123456789ABCDEF") for _ in range(self.nonce_bytes * 2))

    def draw(self) -> SeparatorPair:
        family, open_fmt, close_fmt = self.rng.choice(self.families)
        tag, nonce = self.rng.choice(self.tags), self._nonce()
        return SeparatorPair(family, tag, nonce, open_fmt.format(tag=tag, nonce=nonce), close_fmt.format(tag=tag, nonce=nonce))

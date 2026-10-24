"""Boundary-forgery signals: text that imitates a prompt delimiter or a chat-template control token.

Engine 1 cannot know which marker the attacker will imitate, so this looks for (a) anything shaped like this engine's
own marker families, and (b) well-known role/turn delimiters of common chat templates and "end of input" phrasing.
The result is a signal for the policy engine; Engine 1 never edits the input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("engine_marker", re.compile(r"\[\[/?[A-Z_]{3,12}-[0-9A-F]{6,}\]\]|</?[A-Z_]{3,12}_[0-9A-F]{6,}>|"
                                 r"#{3,}\s*[A-Z_]{3,12}\s+[0-9A-F]{6,}\s+(BEGIN|END)|@@@\s*(BEGIN|END)\s+[A-Z_]{3,12}\s+[0-9A-F]{6,}|"
                                 r"\{\{/?[A-Z_]{3,12}:[0-9A-F]{6,}\}\}|===\[[0-9A-F]{6,}\s+[A-Z_]{3,12}\s+(START|STOP)\]===|"
                                 r"\|~/?[A-Z_]{3,12}~[0-9A-F]{6,}~\||%%%\s*[0-9A-F]{6,}:[A-Z_]{3,12}:(open|close)\s*%%%")),
    ("chat_control_token", re.compile(r"<\|(im_start|im_end|system|user|assistant|endoftext|eot_id|start_header_id|"
                                      r"end_header_id|begin_of_text)\|>|\[/?INST\]|<</?SYS>>|<start_of_turn>|<end_of_turn>",
                                      re.IGNORECASE)),
    ("role_header", re.compile(r"(?m)^\s*(###\s*)?(system|assistant)\s*(prompt|message)?\s*:\s*\S", re.IGNORECASE)),
    ("end_of_input_claim", re.compile(r"(end of (the )?(user )?(input|data|document|email|context)|"
                                      r"(begin|start) (of )?(new|system) (instructions?|prompt))", re.IGNORECASE)),
    ("delimiter_line", re.compile(r"(?m)^\s*[#=*~\-_<>@%|]{3,}\s*(end|begin|start|stop|system|instructions?)\b[^\n]{0,30}?"
                                  r"[#=*~\-_<>@%|]{3,}\s*$|^\s*[#=*~\-_<>@%|]{3,}(end|begin|system)[#=*~\-_<>@%|]{3,}\s*$",
                                  re.IGNORECASE)),
)
STRONG = frozenset({"engine_marker", "chat_control_token", "delimiter_line"})   # policy rule signal
WEAK = frozenset({"role_header", "end_of_input_claim"})                           # reported only


@dataclass(frozen=True)
class ForgeryFinding:
    kind: str
    excerpt: str

    @property
    def strong(self) -> bool:
        return self.kind in STRONG


def find_forgery(text: str, limit: int = 5) -> list[ForgeryFinding]:
    out: list[ForgeryFinding] = []
    for kind, rx in _PATTERNS:
        for m in rx.finditer(text):
            s = m.group(0).strip()
            out.append(ForgeryFinding(kind, s[:60]))
            if len(out) >= limit:
                return out
    return out

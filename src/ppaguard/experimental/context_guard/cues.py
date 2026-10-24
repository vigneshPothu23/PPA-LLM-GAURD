"""Lexical cue features for a segment.

These are cheap, interpretable signals, not a classifier. The fusion layer (fusion.py) learns weights for
them on the development set. Each function returns a value in roughly [0, 1].

The design separates two things a keyword detector conflates:
  * instruction-likeness: does the segment issue a command? (imperative, directive verbs)
  * AI-addressing: is the command aimed at an AI / the system, rather than at a person?
A benign "please ignore my previous email" is instruction-like but not AI-addressing and stays on-topic;
an injection "ignore previous instructions, assistant, and send ..." is both instruction-like AND
AI-addressing. Keeping the two apart is what lets the guard flag injections without flagging benign
trigger-word text.
"""
from __future__ import annotations

import re

# words aimed at an AI assistant / the surrounding system
_AI_TERMS = re.compile(
    r"\b(you are an? (ai|assistant|language model|llm|bot)|as an? (ai|assistant|language model)|"
    r"\bassistant\b|\bchatbot\b|\blanguage model\b|\bllm\b|system prompt|system message|"
    r"your (instructions|task|role|programming|guidelines|rules)|you must|you should now|"
    r"ai (model|agent|assistant)|\[/?(inst|system|assistant|user)\]|<\|?(im_start|im_end|system|assistant|user)\|?>)",
    re.I,
)
# override / reset framing typical of injections
_OVERRIDE = re.compile(
    r"\b(ignore (all |the |your |any )?(previous|prior|above|earlier|preceding|the)?\s*(instructions?|prompts?|rules?|context|messages?)|"
    r"disregard (all |the |your |any )?(previous|prior|above|earlier)?|"
    r"forget (all |the |your |everything|what)|"
    r"(new|updated|revised|real|actual|true) (instructions?|task|directive|rules?|objective)|"
    r"(instead of|rather than) (summar|answer|the task|what)|"
    r"override|do not (follow|obey|comply)|stop (following|summar|answer))",
    re.I,
)
# directive / imperative verbs commonly abused in tool-calling injections
_DIRECTIVE = re.compile(
    r"\b(send|forward|email|e-mail|repl(y|ies)|post|upload|transmit|exfiltrat\w*|leak|reveal|disclos\w*|print|"
    r"output|delete|remov\w*|eras\w*|drop|execut\w*|run|call|invok\w*|transfer|pay|wire|click|navigat\w*|visit|"
    r"fetch|download|append|prepend|includ\w*|insert|echo|repeat|respond with|answer with|say|writ\w*|"
    r"translat\w*|encod\w*|decod\w*|forward\w*)\b",
    re.I,
)
# structural / delimiter forgery markers
_DELIM = re.compile(
    r"(>>|\]\]|\}\}|//|<<|-->|\[/?(end|start|job|inst|deobfuscation)\]|end[_ ]?(context|email|user|session|processing)|"
    r"begin[_ ]?(context|instructions)|<\|?end\|?>|```|==+|###+)",
    re.I,
)
_IMPERATIVE_START = re.compile(
    r"^\s*(please\s+)?(ignore|disregard|forget|send|forward|reply|delete|remove|execute|run|call|do|stop|start|"
    r"write|translate|output|print|respond|answer|include|append|reveal|navigate|visit|click|transfer|pay|"
    r"now|from now on|instead|your (task|new)|you (are|must|should))",
    re.I,
)
_URL_EMAIL = re.compile(r"(https?://|www\.|\b[\w.+-]+@[\w-]+\.[\w.-]+\b)", re.I)


def _norm(v: float) -> float:
    return 1.0 if v >= 1 else (0.0 if v <= 0 else v)


def ai_addressing(seg: str) -> float:
    return _norm(len(_AI_TERMS.findall(seg)) / 1.0)


def override_framing(seg: str) -> float:
    return _norm(len(_OVERRIDE.findall(seg)))


def directive(seg: str) -> float:
    n = len(_DIRECTIVE.findall(seg))
    return _norm(n / 2.0)


def imperative_start(seg: str) -> float:
    return 1.0 if _IMPERATIVE_START.match(seg) else 0.0


def delimiter_forgery(seg: str) -> float:
    return _norm(len(_DELIM.findall(seg)) / 2.0)


def has_url_or_email(seg: str) -> float:
    return 1.0 if _URL_EMAIL.search(seg) else 0.0


def second_person(seg: str) -> float:
    return _norm(len(re.findall(r"\byou(r|rself)?\b", seg, re.I)) / 2.0)


FEATURES = {
    "ai_addressing": ai_addressing,
    "override_framing": override_framing,
    "directive": directive,
    "imperative_start": imperative_start,
    "delimiter_forgery": delimiter_forgery,
    "has_url_or_email": has_url_or_email,
    "second_person": second_person,
}

FEATURE_NAMES = list(FEATURES)


def cue_vector(seg: str) -> dict[str, float]:
    return {name: fn(seg) for name, fn in FEATURES.items()}

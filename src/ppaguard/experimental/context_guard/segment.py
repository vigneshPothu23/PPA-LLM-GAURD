"""Channel-aware segmentation of untrusted document text.

An indirect prompt injection is a short span inserted into otherwise ordinary content. Scoring the whole
document lets a short injection be diluted and lets a benign trigger word in a clean document dominate.
The context guard instead scores segments and aggregates.

Segmentation is format-aware:
  * tables (Markdown-style `a | b | c` rows) -> each cell becomes a segment, tagged "table_cell";
  * code (lines beginning with comment markers, or fenced blocks) -> each comment becomes a segment,
    tagged "code_comment"; non-comment code lines are tagged "code" and down-weighted;
  * prose -> sentences, tagged "sentence"; a very long sentence is split on newlines first.

Each segment keeps its character offsets so an explanation can point back into the document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_COMMENT = re.compile(r"^\s*(#|//|/\*|\*|--)\s?(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|?[^|\n]*\|[^|\n]*\|?\s*$")


@dataclass
class Segment:
    text: str
    kind: str          # sentence | table_cell | code_comment | code | line
    start: int         # char offset into the original document (approximate for derived cells)
    end: int


def _looks_like_table(block: str) -> bool:
    lines = [l for l in block.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    piped = sum(1 for l in lines if l.count("|") >= 1)
    return piped >= max(2, int(0.6 * len(lines)))


def _split_sentences(text: str, base: int) -> list[Segment]:
    out: list[Segment] = []
    pos = 0
    for raw in _SENT.split(text):
        s = raw.strip()
        if not s:
            pos += len(raw) + 1
            continue
        idx = text.find(s, pos)
        if idx < 0:
            idx = pos
        # split an over-long "sentence" on newlines so inserted lines separate out
        if len(s) > 400 and "\n" in s:
            sub = 0
            for part in s.split("\n"):
                p = part.strip()
                if p:
                    j = s.find(p, sub)
                    out.append(Segment(p, "line", base + idx + max(0, j), base + idx + max(0, j) + len(p)))
                    sub = j + len(p)
        else:
            out.append(Segment(s, "sentence", base + idx, base + idx + len(s)))
        pos = idx + len(s)
    return out


def segment(text: str) -> list[Segment]:
    if not isinstance(text, str) or not text.strip():
        return []
    segs: list[Segment] = []
    # split the document into blocks on blank lines, so a table or code block is handled as a unit
    blocks = re.split(r"\n\s*\n", text)
    offset = 0
    for block in blocks:
        bstart = text.find(block, offset)
        if bstart < 0:
            bstart = offset
        offset = bstart + len(block)
        stripped = block.strip()
        if not stripped:
            continue
        if _looks_like_table(block):
            for line in block.splitlines():
                if not line.strip() or set(line.strip()) <= {"-", "|", " ", ":"}:
                    continue
                cells = [c.strip() for c in line.split("|")]
                for c in cells:
                    if c:
                        idx = text.find(c, bstart)
                        segs.append(Segment(c, "table_cell", idx if idx >= 0 else bstart,
                                            (idx if idx >= 0 else bstart) + len(c)))
            continue
        comment_lines = 0
        code_like = sum(1 for l in block.splitlines() if _COMMENT.match(l) or
                        re.search(r"[;{}()=]|def |class |SELECT |import |function ", l))
        if code_like >= max(2, int(0.5 * len([l for l in block.splitlines() if l.strip()]))):
            for line in block.splitlines():
                if not line.strip():
                    continue
                m = _COMMENT.match(line)
                idx = text.find(line.strip(), bstart)
                if m and m.group(2).strip():
                    comment_lines += 1
                    segs.append(Segment(m.group(2).strip(), "code_comment",
                                        idx if idx >= 0 else bstart,
                                        (idx if idx >= 0 else bstart) + len(m.group(2).strip())))
                else:
                    segs.append(Segment(line.strip(), "code",
                                        idx if idx >= 0 else bstart,
                                        (idx if idx >= 0 else bstart) + len(line.strip())))
            continue
        segs.extend(_split_sentences(block, bstart))
    # drop empties, keep order
    return [s for s in segs if s.text.strip()]

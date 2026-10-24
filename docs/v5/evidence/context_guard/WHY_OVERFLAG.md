# Why the whole-document classifiers over-flag (and miss embedded injections)

This note records the diagnosis that motivated the context guard. All numbers are measured, with the
command that produced them. It is a working note; the headline comparison is in `../../CONTEXT_GUARD.md`.

## Two failure modes of a single whole-document classifier

A prompt-injection classifier is trained on short, mostly self-contained injection strings and benign
prompts. An LLM application instead feeds it a whole e-mail, document or table. Two things then go wrong:

### 1. Dilution — embedded injections are missed

When a short injected instruction is placed inside a larger benign document, the document is mostly benign
text. Scoring the whole thing (even with windowing + max) tends to wash out the signal.

Measured on the development set `eval/context_sets/ctx_dev.jsonl` with the default detector
(`protectai/deberta-v3-base-prompt-injection-v2`, whole-document inference):

| Content type | n | mean P(injection) | flagged at 0.5 |
|---|---|---|---|
| clean benign documents | 38 | 0.001 | 0 / 38 |
| benign documents with trigger words | 30 | 0.109 | 3 / 30 |
| benign documents with an **embedded injection** (38 published injection segments + 8 harmless off-task requests) | 46 | 0.251 | 11 / 46 |

The embedded injections (TaskTracker-style topic-shift directives) score only 0.25 on average and are
caught less than a quarter of the time at 0.5. The classifier is not blind to them, but at the document
level the evidence is too weak to act on.

### 2. Trigger-word over-firing — benign content is flagged

On the frozen test set, the NotInject subsets are benign prompts that deliberately contain words such as
"ignore", "password", "system" and "delete". These carry no instruction aimed at an AI, yet the
whole-document classifiers flag a large fraction of them (see `../../EVALUATION.md`: ProtectAI flags
54–60%, deepset 50–88%). A keyword-reactive model cannot tell "please **ignore** my previous e-mail"
(a human talking to a human) from "**ignore** your previous instructions" (an attacker talking to the model).

## The design consequence

Both failures come from collapsing a structured document into one score. The two cases a single score
cannot separate are:

- *instruction-like but benign*: on-topic, addressed to a person — should be ALLOW;
- *instruction-like and hostile*: off-topic, addressed to the model, or forging structure — should WARN/BLOCK.

The context guard therefore (a) scores **segments**, not the whole document, so an embedded injection is
not diluted; and (b) adds features that a keyword model lacks — whether a segment is **off-topic** relative
to its document, and whether it **addresses an AI / forges structure** — so that benign trigger words do
not, on their own, raise the score. Whether this actually lowers the false-positive rate while keeping
recall is measured in `../../CONTEXT_GUARD.md`; it is not assumed here.

## Reproduce

```
python scripts/context_guard/build_datasets.py
python scripts/context_guard/cache_features.py --set ctx_dev
# the dilution table above = the cached whole-document ProtectAI scores ("wholedoc.protectai" in
# docs/v5/evidence/context_guard/cache/ctx_dev.jsonl), grouped by the attack_style field of ctx_dev.jsonl
```

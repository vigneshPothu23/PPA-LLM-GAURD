# Engine 2 experiment: does input handling fix BIPIA without fine-tuning?

**Status (2026-10-21): DECISION PENDING.** The corrected experiment is built and verified, but it has **not been run on the real ProtectAI model**. The build sandbox's network policy refuses huggingface.co (HTTP 403), so the weights were never available there. Every metric printed by the dry run came from a random stub model and is not a result.

## What was fixed (verified)
| v1 problem (`dev_indirect.jsonl`, 44,250 rows) | v2 (`dev_indirect_v2.jsonl`, 988 rows) |
|---|---|
| All rows label `"1"` (string); no clean rows | 247 clean (label 0) + 741 poisoned (label 1), integer labels |
| 2 email contexts also in BIPIA **test** | 0 rows contain any BIPIA test context or test-pool attack; 0 exact/normalised overlap with frozen validation/test |
| Row-level holdout: 163 unique tables, 3,825 duplicate rows | Unique contexts only; 0 duplicates |
| Every attack × every context (91.5 % tables) | 1 attack per context × position, rotated over categories (email 15, table 15, code 10 categories) |
| 10 email / 10 code contexts | All leak-free BIPIA-train contexts: email 34, code 50; table 163 (v1 holdout) |

v2 SHA-256: `71d53ad32c419e91…` (full value in `data/dev_indirect_v2.jsonl.build_log.json`). Rebuilding gives the same hash.

| v1 probe bug | v2 probe |
|---|---|
| HEAD_TAIL = first 250 + last 250 **words** → duplicated text in 44,025/44,250 rows | Only when > 510 content **tokens**: first 128 + last 382 tokens, sliced by tokenizer offsets (LLM Guard defaults); otherwise identical to FULL |
| CHUNKS = 350-word split/join → newlines removed, mostly 1 chunk, may still exceed 512 tokens | LLM Guard `CHUNKS` reproduced exactly: 256-character slices of the original string, 25-char overlap, max score; matches the reference function on test lengths |
| `raw_score > 0.92` | Official rule: `round(P, 2) > 0.92` |
| "Recall" = flagged / all rows; no per-row output | Recall, FPR, precision, F1 per task and position, Wilson 95 % CIs, exact McNemar vs FULL, per-row scores, data and model hashes |

Checks: `experiments/tests/test_probe_v2.py` (18 pass). The project suite still passes (126 passed, 3 skipped that need real weights). End-to-end dry run on the stub: 1,647 rows (988 dev + 659 frozen validation) in about 13 s. The frozen test set is never opened.

## Pre-registered decision rule (in the script header, fixed before any real result)
A strategy **helps** only if, against FULL, all of these hold:
- pooled poisoned recall rises by at least 0.15, with McNemar p < 0.01 and non-overlapping 95 % CIs;
- clean-context FPR rises by no more than 0.02 pooled and no more than 0.05 per task;
- frozen-validation deepset recall does not drop, and benign FPR rises by no more than 0.02.

**DO NOT FINE-TUNE** if a helping strategy reaches recall ≥ 0.50 with clean FPR ≤ 0.10. If one helps but stays below that bar, the result is **PARTIAL → FINE-TUNE**. Otherwise **FINE-TUNE**.

## Run (one command, CPU, est. 30–40 min at your measured 358 ms per FULL call)
```
python experiments/scripts/option_c_probe_v2.py --model-dir <folder containing onnx/> --out experiments/results/option_c_v2
```
Outputs: `REPORT.md`, `metrics.json` (with the decision), and `per_row_scores.jsonl`.

## Deployability of any fine-tuned model
A fine-tune of the same checkpoint has the identical architecture (184,423,682 parameters; ~738 MB ONNX). A same-architecture stand-in measured 1.28 GB RAM for the whole app, within Streamlit Community Cloud's documented 2.7 GB maximum. A fine-tuned model must be exported with the same `onnx/` layout and given new thresholds chosen on validation.

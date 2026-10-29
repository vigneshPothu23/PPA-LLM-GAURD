# PPA Guard: final evaluation (audit, fix, one-time holdout)

Everything below was measured in this repository with the scripts named in each section. Nothing is estimated or
typed in by hand from memory. Synthetic data is labelled as synthetic. The application is a prompt-protection and
scanning system: it does **not** call an LLM, so no attack-success rate against a downstream model is reported.

## 1. What was wrong (measured on development data only)

The shipped application (`main` 247c9f8) runs ProtectAI DeBERTa-v3 v2 on the whole input (512-token windows, max
over windows) with WARN > 0.01 and BLOCK > 0.99. Two root causes were found on the **development** split of the
security suite and on the development pool; no holdout row was inspected.

**RC1 - dilution.** A short instruction inserted into a longer benign document is scored together with the whole
document, and the score collapses:

| Development case | Whole document | The injected line alone | Baseline decision |
|---|---|---|---|
| README with an instruction in an HTML comment (`d7264930c518`) | 0.0012 | 1.0000 | ALLOW (forwarded) |
| E-mail with an embedded instruction (`3223560077af`) | 0.0064 | 0.9999 | ALLOW (forwarded) |

**RC2 - trigger-word over-firing and a miscalibrated WARN threshold.** WARN > 0.01 is far below the detector's
operating range, and ordinary text with words such as "ignore", "secret", "override" is scored near 1.0. Example:
a benign e-mail asking colleagues to ignore an earlier message (`ea7fc9cb98aa`) scored P = 1.0 and was BLOCKED.

Simply scoring every line separately fixes RC1 but makes RC2 worse (every benign line with a trigger word fires).

## 2. What changed

A context signal was added to Engine 2 and the thresholds were re-selected. Design and thresholds were chosen on
development data only and pre-registered before any holdout row was scored
(`docs/v5/evidence/final/PREREGISTRATION_FINAL.md`; set hashes verified unchanged before scoring).

- **Context signal** (`src/ppaguard/engine2/context/`): segments (sentences, table cells, code comments; over-long
  text split into lines) → 10 features (7 lexical cues for instructions addressed to an AI or forging a boundary;
  the Engine 2 detector on the segment; off-topic distance in MiniLM-L6 embeddings; segment kind) → a logistic
  regression with 10 weights fitted earlier on development data (round 1 of the context-guard research,
  `artifacts/context_guard/guard_full.json`, unchanged). Document score = highest segment score.
  **No model was trained or fine-tuned in this phase**; the transformer weights are ProtectAI's and MiniLM's
  published checkpoints at pinned revisions.
- **Policy:** BLOCK if whole-document P > 0.99 (unchanged); **WARN if context P > 0.95** (WARN only; held for
  analyst review); the whole-document WARN band was removed (the selected value 0.999 lies above BLOCK; stored as
  WARN = BLOCK = 0.99, which yields identical decisions). Rule signals unchanged.
- **Fail closed:** detector and context signal load as one unit with hash-verified files; a missing or corrupt
  model, or an error during a scan, BLOCKs.

## 3. How the design was chosen (development data only)

Five candidate designs were compared by nested, grouped 5-fold cross-validation on the pooled development data
(frozen_dev, ctx_dev, dev2_email, bipia_train_dev, scenarios_dev: 2,325 rows, 1,343 injected; groups = source
document or carrier, so variants of one document never straddle folds). Thresholds were re-selected inside every training fold
with the project's FPR-bounded rule (WARN: maximise recall with FPR ≤ 10 %; BLOCK: precision ≥ 0.98 and FPR ≤ 1 %,
else 0.99). Pre-stated criterion: highest out-of-fold Youden index (recall − FPR).
Script: `scripts/context_guard/select_design.py`; output `docs/v5/evidence/final/design_selection_cv.json`.

| Variant | TP | FP | FN | TN | Recall | FPR | Precision | F1 | Youden [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|
| A shipped (whole doc, WARN 0.01) | 572 | 212 | 771 | 770 | 0.426 | 0.216 | 0.730 | 0.538 | 0.210 [0.171, 0.248] |
| B whole doc, re-selected thresholds | 357 | 98 | 986 | 884 | 0.266 | 0.100 | 0.785 | 0.397 | 0.166 [0.135, 0.198] |
| C max(whole doc, per segment) | 565 | 112 | 778 | 870 | 0.421 | 0.114 | 0.835 | 0.559 | 0.307 [0.274, 0.339] |
| **D whole doc OR context signal (selected)** | **788** | **140** | **555** | **842** | **0.587** | **0.143** | **0.849** | **0.694** | **0.444 [0.409, 0.478]** |
| E as D, context model re-fitted per fold | 513 | 112 | 830 | 870 | 0.382 | 0.114 | 0.821 | 0.521 | 0.268 [0.234, 0.303] |

Threshold-free ROC-AUC on the same pool: whole document 0.728, per-segment max 0.780, context signal 0.848.

Known costs already visible on development data (reported before the holdout): on real e-mail with attacker-written
text (dev2_email) the FPR of D rose from 0.217 (A) to 0.372; the BLOCK precision target was not met (BLOCK precision
0.84, FPR 0.054 on dev), so BLOCK stays at the 0.99 fallback.

The context model was fitted in round 1 on `ctx_dev`, which is part of this pool, so D's out-of-fold numbers on
`ctx_dev` are optimistic; E (re-fitted inside each fold) is the leakage-free control and scored lower. This is why
the decision rests on the holdout sets below, not on the development numbers.

<!-- RESULTS -->

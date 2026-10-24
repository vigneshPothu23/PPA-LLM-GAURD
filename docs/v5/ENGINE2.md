# Engine 2: transformer prompt-injection detector (v5, independent implementation)

Code: `src/ppaguard/engine2/`. Tests: `tests/v5/test_v5_engine2.py` (runtime, data, metrics, registry) and
`tests/v5/test_v5_training.py` (the whole pipeline, reproducibility, leakage refusal, overwrite protection).

| Module | Responsibility |
|---|---|
| `data.py` | Reads JSONL/JSON/CSV. Validates each row (text type, empty, NUL, label). Normalises labels. Removes exact (normalised) duplicates and every row of a text that carries both labels. Splits by group and label, keeping paired clean/poisoned variants together. **Refuses** training data that shares any normalised text with the frozen dev or test sets. |
| `tokenization.py` | Loads the model's own `tokenizer.json`. Checks it against the model: vocabulary ≤ embedding rows, required special tokens, ids in range. Reports token-length statistics. |
| `train.py` | The pipeline. Configuration from JSON → data → tokenizer → model → training (class-weighted cross-entropy, AdamW, warm-up + linear decay, gradient clipping, length-bucketed batches, evaluation on an internal split, early stopping, time budget) → best checkpoint → ONNX export with a parity check against PyTorch → optional int8 quantization with a quality check → temperature calibration and threshold selection on the **frozen dev** set → **one** evaluation on the frozen test set → model card with file hashes. |
| `metrics.py` | numpy only: confusion matrix, per-class precision/recall/F1, ROC-AUC (tie-aware), average precision, Brier score, ECE, NLL, temperature fitting, threshold selection. |
| `runtime.py` | Inference with ONNX Runtime on CPU (no torch). Verifies file hashes from the model card at load. Scores long text in overlapping windows and takes the maximum, flagging `partial` when there are more windows than the budget. Applies the calibrated temperature. Gives occlusion explanations. Clear `ModelUnavailable` and `InferenceError`. |
| `registry.py` | Chooses the model: `PPAGUARD_MODEL_DIR`, or `PPAGUARD_MODEL=protectai` (the default; pinned download of `protectai/deberta-v3-base-prompt-injection-v2`), or `PPAGUARD_MODEL=<name>` for a bundled artifact. There is no fallback: anything unavailable gives `ModelUnavailable`, and the workflow BLOCKs. |
| `evaluate.py` | Evaluates any model folder on labelled data with the production inference code. |

## Data discipline

- **Training data:** `experiments/results/injecguard_audit_v2/candidate_train.jsonl`, the audited v2 candidate (71,201
  rows, SHA-256 `44683068…ec34`). The earlier audit already removed BIPIA, over-defence and dev-source rows, duplicates,
  conflicts and leakage. The pipeline repeats its own leakage check and stops if it finds any overlap.
- **Early stopping** uses a 5% group-aware split of the *training* data.
- **The frozen dev set** (`eval/sets/validation.jsonl`, 659 rows) sets the temperature and the WARN/BLOCK thresholds.
- **The frozen test set** (`eval/sets/test.jsonl`, 1,026 rows: BIPIA indirect injections and NotInject over-defence
  probes) is evaluated **once**, after every choice has been fixed. Nothing reads it earlier.

## Threshold rule (on the dev set)

- **BLOCK:** the lowest threshold whose injection precision is at least 0.98 there *and at every higher threshold*.
  This keeps false BLOCKs rare and avoids picking an isolated lucky point.
- **WARN:** the highest threshold, at or below BLOCK, that still flags at least 95% of injections.
- If a target cannot be met, the rule falls back to conservative values and records a note.
- Applications can override the thresholds. The UI then labels them "not validated".

## Explanations

The explanation is occlusion: each sentence (or word span) is removed in turn, and the drop in P(injection) is
recorded. The segments with the largest drops are shown. Every number comes from the same model. These are
approximate attributions, not proofs.

## Model options

| Option | Provenance label | When |
|---|---|---|
| `PPAGUARD_MODEL=protectai` (default) | `pretrained-external`, EXPERIMENTAL | ProtectAI's published classifier at a pinned revision. Thresholds WARN 0.01 / BLOCK 0.99 selected on the frozen dev set. Dev ROC-AUC 0.823, test ROC-AUC 0.396. |
| `PPAGUARD_MODEL=roberta-6l-experimental` | `fine-tuned-ppaguard`, EXPERIMENTAL | RoBERTa-base (spaCy en_core_web_trf 3.8.0 weights, MIT, conversion verified), 6 layers, fine-tuned here. int8 ONNX, 83 MB. Dev 0.716, test 0.547. |
| `PPAGUARD_MODEL=scratch-baseline` | `trained-from-scratch-ppaguard`, EXPERIMENTAL | 4-layer BERT trained from random initialisation, 7.7 MB. Dev 0.658, test 0.660. |
| `configs/engine2/finetune_deberta_v3_xsmall.json` | `fine-tuned-ppaguard` | Ready to run with Hugging Face access. Not run here; see EVALUATION.md for why. |

The application always shows which of these is running: the sidebar, the Scan Console, the System page and the
Engine 2 Model page.


## Results

See [EVALUATION.md](EVALUATION.md) for the full comparison: six candidates, dev/test, per-subset results, contamination
flags, quantization and threshold notes. In short, no detector is validated for production on the frozen test set.

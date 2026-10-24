# PPA Guard v5: build report (2026-10-24)

## Verification stages (kept separate)

| Stage | Result | Evidence |
|---|---|---|
| Unit and integration tests, clean Python 3.12 environment from `requirements.txt` | **485 passed, 4 skipped.** The suite runs on SQLite and PostgreSQL 16. 1 skip is the training test (no torch in the runtime environment); 3 are legacy real-model tests, which were then run separately (below). | `docs/v5/evidence/test_run_clean312.txt` |
| Training-pipeline tests (torch 2.14.1, transformers 5.20.1) | **5 passed.** Covers end to end, reproducibility, leakage refusal, no overwrite, recalibration. | `docs/v5/evidence/test_run_training.txt` |
| Legacy real-model tests with the real ProtectAI checkpoint | **3 passed.** These were always skipped before Hugging Face access. | `docs/v5/evidence/test_run_legacy_real_model.txt` |
| Real model training, executed here | Two runs: from scratch (58 min) and RoBERTa-6L fine-tune (109 min). | `docs/v5/evidence/engine2/*/training_log.jsonl` |
| Detector comparison on the frozen sets | Six candidates, real weights, production inference code. | `docs/v5/EVALUATION.md`, `docs/v5/evidence/engine2/comparison/` |
| Local app, browser walkthrough (Chromium, Playwright) with the default detector | **21/21 checks passed**; 0 JavaScript errors; peak RSS **1,355 MB**. | `docs/v5/evidence/walkthrough/` (log and 11 screenshots) |
| GitHub push | See the section below. | |
| Streamlit Community Cloud deployment | **Not deployed by me.** No Streamlit account authorisation is available in this session. | |

## Outcomes against the brief

1. **Independent implementation.** `src/ppaguard/` holds new Engine 1, Engine 2, policy, core, SDK and UI.
   `test_v5_namespace_never_imports_the_old_implementation` scans the code for imports of the old package, and a
   second test runs the workflow with the old package blocked from import. Both pass.
2. **Security workflow.** Sign-in → role, organization and application scope → rate limit → input validation →
   Engine 2 → Engine 1 forgery signal → policy → Engine 1 assembly (when forwarding) → one transactional record plus
   audit event → result. Every failure path is BLOCK (tested).
3. **ALLOW / WARN / BLOCK.** Tested in `test_v5_policy.py` and `test_v5_core.py`, and through the UI. A release never
   overrides a BLOCK, is single-use, triggers a full re-check, expires, and is safe under concurrent requests.
4. **Five-role RBAC** is enforced in the backend, with parametrised tests for unauthorized access.
5. **Integrations.** The Streamlit console and `ppaguard.sdk.PPAGuardClient`. `examples/integration_example.py` is run
   by the tests and writes `sdk`-channel records.
6. **Real machine learning.**
   - Legitimate weights: ProtectAI v2 (Apache-2.0), deepset DeBERTa (MIT), fmops DistilBERT (Apache-2.0), and
     RoBERTa-base from spaCy `en_core_web_trf` 3.8.0 (MIT; conversion proven equal on hidden states and token ids).
   - Each model uses its own tokenizer.
   - Training and inference results are reported only where they were executed.
   - **No detector is validated for production** (see below).
7. **UI.** Teal and charcoal theme. The main workflows were exercised in a real browser.
8. **Zero cost.** No paid API, inference service, hosting or database is required. The default model is a free
   download.
9. **Regression protection.**
   - `src/ppa_guard/`, `app_pages/`, `legacy/`, `eval/`, `experiments/` and the frozen sets are unchanged. Frozen-set
     SHA-256 values: dev `eef2da60…`, test `e00b70c3…`.
   - The legacy suites still pass.
   - Nothing was deleted.
10. **Reproducibility.**
    - Pinned `requirements*.txt` files and `configs/engine2/*.json`.
    - `scripts/convert_spacy_roberta.py`, `scripts/prepare_hf_classifier.py` and `scripts/compare_detectors.py`.
    - Seeds, data hashes and environment versions are recorded in every model card. See `docs/v5/DEPLOY.md`.
11. **Honest report.** This document and `EVALUATION.md`.

## Engine 2: actual model status

| Model | Dev ROC-AUC | Test ROC-AUC | Test injection P / R / F1 / FPR at its dev-selected WARN threshold |
|---|---|---|---|
| ProtectAI v2 (**default, EXPERIMENTAL**) | 0.823 | 0.396 | 0.458 / 0.320 / 0.377 / 53.3% |
| roberta6l int8 (ours) | 0.716 | 0.547 | 0.614 / 0.493 / 0.547 / 43.7% |
| scratch (ours) | 0.658 | 0.660 | 0.686 / 0.862 / 0.764 / 55.6% |
| fmops DistilBERT (dev-contaminated) | – | 0.808 | at 0.5: 0.631 / 1.000 / 0.774 / 82.4% |
| deepset DeBERTa (dev-contaminated) | – | 0.586 | at 0.5: 0.628 / 1.000 / 0.771 / 83.6% |

All values come from `docs/v5/evidence/engine2/comparison/*/comparison.json`. Full tables are in `EVALUATION.md`.

**Conclusion.** None of these detectors works as a stand-alone production control on this test distribution.
- Instructions hidden inside e-mails, tables and code (BIPIA) are not separated from clean documents.
- Harmless trigger-word prompts are often flagged.
- The application is honest about this. Engine 2 is labelled EXPERIMENTAL in the sidebar, the Scan Console, the model
  card and the Engine 2 Model page. Engine 1, the rule signals and analyst review remain in place.

## Issues found and fixed during this phase

- **Threshold rule.** The recall-first rule produced degenerate thresholds for weak models. It was replaced by an
  FPR-bounded rule, disclosed in the model cards, with a re-scored test set for the affected model.
- **BLOCK fallback note.** The note text misstated the false-positive rate when even 0.99 exceeded the budget. Fixed.
- **ONNX export.** Under transformers 5.x, tracing with an all-ones mask dropped attention masking for DeBERTa-v2 and
  DistilBERT. The exporter now traces with a padded batch and keeps the model in eval mode. Parity and padding
  invariance are checked on every export.
- **Paused applications** now also refuse API-key calls in v5. The v4 platform layer in `src/ppa_guard/platform/` still
  has this gap and was left unchanged, as instructed.

## Remaining blockers

1. **Streamlit deployment** needs your Streamlit account, at share.streamlit.io. Use repository
   `vigneshPothu23/PPA-LLM-GAURD`, branch `main`, main file `streamlit_app.py`, Python 3.12, and secrets as in
   `docs/v5/DEPLOY.md` §5. I cannot open or verify a public URL until it exists.
2. **A production-grade detector** needs training data that covers indirect, in-document instructions and
   over-defence negatives, from sources other than the frozen sets, and validation on a fresh held-out set.
3. **Gated models**, e.g. ProtectAI deberta-v3-small and Meta Prompt Guard, need a Hugging Face token and licence
   acceptance.

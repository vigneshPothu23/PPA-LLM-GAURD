# Pre-registration: final submission evaluation (written before any holdout was scored)

Written 2026-10-25T11:26:41Z. At this time no feature, score or decision had been computed for
`scenarios_holdout` or `final_holdout` by any configuration, including the shipped baseline. The baseline was scored
on the whole scenario file (dev + holdout) during Phase A reproduction. Only its **dev split** was inspected for root
causes; holdout rows were not read or analysed.

## Selected configuration (design D, chosen by nested grouped 5-fold CV on development data only)

`docs/v5/evidence/final/design_selection_cv.json` holds the five candidates and their out-of-fold numbers.

- Engine 2a: whole-document ProtectAI v2 (pinned revision `89b085cd`). **BLOCK if P > 0.99.** (The selected
  whole-document WARN threshold, 0.999, falls above BLOCK, so the whole-document score adds no WARN band.)
- Engine 2b: the round-1 context guard, unchanged artifact `artifacts/context_guard/guard_full.json`
  (SHA-256 `dc384dda97cd79f249ffc3f19d11ed03597bc5e576d383eed6e1eff02d201cea`). **WARN (held for review) if P > 0.95.** The guard alone never BLOCKs.
- Policy rules unchanged (input validation, Engine 1 boundary forgery → WARN, hidden characters → WARN, partial
  analysis → WARN, fail closed on any error). Most severe signal wins.
- Thresholds selected on the pooled development data: frozen_dev, ctx_dev, dev2_email, bipia_train_dev and
  scenarios_dev (2,325 rows, 1,343 injected). Rule: WARN maximises recall with FPR ≤ 10 %; BLOCK uses the project's
  precision rule, falling back to 0.99 because the target was not met.

## Holdout sets (each scored once, with both the baseline and the final code, through `eval/run_pipeline_eval.py`)

| Set | SHA-256 | Rows (injected) | Kind |
|---|---|---|---|
| scenarios_holdout | `292be00ca52770315b3f15e724d4c68583b2c0886bc7896b7854991091c4be7a` | 84 (53) | synthetic, hand-written; disjoint carriers and payloads from the dev split |
| final_holdout | `963ed285508332752a8c9aa03aef79c0924be69838b42c2026be574b9bf45d86` | 800 (400) | real: LLMail-Inject phase-2 attacks, fresh Enron e-mail (new byte range), Enron with embedded published injections |
| frozen test | `e00b70c3…9266` | 1026 (600) | existing protocol set; round 1's guard design had seen it, so it is reported as development-influenced |

## Hypotheses (final code vs shipped baseline `main` 247c9f8, same harness, same inputs)

- **F1:** Youden index (recall − FPR) on scenarios_holdout: final > baseline.
- **F2:** Youden index on final_holdout: final > baseline.
- **F3:** Youden index on the frozen test: final > baseline (development-influenced; reported, not decisive).

All of recall, FPR, precision, F1, the confusion matrix, per-category results, blocked-benign counts and latency are
reported whatever the outcome. Results that contradict these hypotheses are reported as such.

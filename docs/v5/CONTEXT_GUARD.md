# Context guard (EXPERIMENTAL): context-aware detection of indirect prompt injection

> **Status: experimental, not validated for production, not wired into the application.** The shipped app,
> its default detector (ProtectAI v2) and its policy are unchanged. The guard lives in
> `src/ppaguard/experimental/context_guard/` and nothing in `core/`, `sdk.py` or `ui/` imports it (a test checks this).

Related: [threat model](THREAT_MODEL.md) · [why whole-document classifiers over-flag](evidence/context_guard/WHY_OVERFLAG.md) ·
[full generated comparison](evidence/context_guard/comparison.md) · [pre-registration](evidence/context_guard/PREREGISTRATION.md) ·
[Engine 2 baseline evaluation](EVALUATION.md)

## 1. Result in brief

<!-- SUMMARY:BEGIN -->
**The guard is better on one class of attack and worse on another. It does not replace the current default.**

- **Better: instructions hidden inside documents, and benign trigger-word text.** On the frozen test set (BIPIA
  hidden e-mail/table/code injections + NotInject trigger-word prompts) its ROC-AUC is **0.913 [0.89–0.93]**. The
  best existing detector reaches 0.808 [0.78–0.84] (fmops) and the current default 0.396 (ProtectAI). At its ctx_dev
  WARN threshold it flags **3.8%** of benign rows, against ProtectAI's 47%, with slightly higher recall (0.36 vs
  0.28). It flags **0%** of the NotInject trigger-word prompts, against 53–58% for ProtectAI.
- **Worse: whole e-mails written by an attacker.** On both independent e-mail sets (LLMail-Inject attacks against
  real Enron and LLMail benign mail) it is clearly worse than the current default. ROC-AUC is 0.70 and 0.58 against
  ProtectAI's 0.92 and 0.90. Recall at WARN is 0.48 and 0.33 against 0.67 and 0.68, and the FPR on real e-mail is
  higher (10–12% vs 8%). The threat model predicted this weakness: an e-mail that *is* the attack has no benign
  context for the off-topic feature to contrast against.
- **The post-hoc fix failed its pre-registered test.** `hybrid_max` (the maximum of guard and ProtectAI) did not
  reach ProtectAI's recall on the confirmatory set (0.61 vs 0.68) and raised the FPR (0.13).
- **No detector, old or new, combines high recall with a low false-positive rate across all content types.** The
  guard is therefore kept as an experimental, review-only signal. It is not proposed as the production detector.
<!-- SUMMARY:END -->

## 2. The problem it addresses

The six whole-document classifiers evaluated earlier fail in two opposite ways
([details](evidence/context_guard/WHY_OVERFLAG.md)):

1. **Dilution.** A short instruction hidden inside a long e-mail, table or code file is mostly surrounded by benign
   text, so a whole-document score stays low. ProtectAI ranks the frozen test's BIPIA documents *worse than chance*
   (ROC-AUC 0.396).
2. **Trigger-word over-firing.** Benign text containing "ignore", "password", "system" or "delete" is flagged although
   nothing in it addresses an AI. ProtectAI flags 53–58% of the NotInject trigger-word prompts in the frozen test.

## 3. Design: three layers

| Layer | What it does | Changed? |
|---|---|---|
| **Engine 1: protection** | Per-request random boundaries isolate untrusted content from trusted instructions. `find_forgery` reports attempts to fake those boundaries (chat-control tokens, delimiter lines, role headers). | No. Reused as-is. Its forgery findings feed the policy. |
| **Context guard: smart detection** | Splits the document into typed segments (sentences, table cells, code comments, lines). Scores each segment on (a) lexical cues that separate *instruction-like* from *addressed to an AI / forging structure*, (b) the existing Engine 2 detector run **on the segment** rather than the whole document, and (c) an **off-topic** score: how unrelated the segment is to the rest of its document (MiniLM sentence embeddings). A small logistic regression fuses the features, and the document score is the maximum over segments. | New, `experimental/context_guard/` |
| **Policy: decision** | The guard's score enters the *existing* `policy.evaluate` together with Engine 1's forgery findings, so input validation, hidden-character and forgery rules, fail-closed behaviour and "max severity wins" all apply. **Conservative:** the guard alone can raise at most a WARN that is *held for analyst review*. It never BLOCKs on its own, because it missed the BLOCK precision target (0.98) on development data. | Policy unchanged. Adapter: `policy_bridge.py` |

Footprint: about 630 lines of numpy / ONNX Runtime code and no new runtime dependency. Extra model: MiniLM-L6
(Apache-2.0, 90 MB ONNX). CPU cost is bounded by at most 16 localised detector calls per document. Measured cache
throughput on 2 cores was about 0.2–0.4 s per document for the full feature set, including three whole-document
baselines.

**Every component is justified on development data.** Five-fold cross-validation on ctx_dev:

| Variant | ctx_dev out-of-fold ROC-AUC |
|---|---|
| full guard | 0.821 |
| without the localised detector | 0.795 |
| without the off-topic feature | 0.739 |
| lexical cues only | 0.716 |
| ProtectAI, whole document | 0.623 |

The full guard was the pre-planned primary configuration. Cross-validation supports keeping all three feature groups.

## 4. Data and leakage control

| Set | Use | Contents | Rows (injected) |
|---|---|---|---|
| `ctx_dev` | **fitting and thresholds only** | Pool A carriers (e-mails, documents, tables, code written for this project), published TaskTracker injected segments (hash split: *held-in* half), pool A benign trigger-word sentences, harmless off-task requests | 114 (46) |
| frozen dev | reported, never tuned on by the guard | deepset direct prompts + NotInject | 659 (203) |
| frozen test | reported | BIPIA e-mail/table/code + NotInject | 1,026 (600) |
| `unseen_synth` | generalisation | Pool B carriers (disjoint), *held-out* TaskTracker segments, pool B trigger-word sentences | 97 (35) |
| `unseen_email` | generalisation, **independent** | LLMail-Inject phase-1 attack e-mails (MIT), LLMail benign FP-test e-mails, real Enron e-mails (Apache-2.0 repackaging) | 961 (398) |
| `unseen_email_2` | **confirmatory, pre-registered, scored once** | LLMail-Inject phase-2 attacks + Enron e-mails from disjoint byte ranges | 800 (400) |

- **No attack text was written for this project.** Injected content comes from published datasets. Pool A and B
  contain only benign carriers, benign trigger-word sentences and harmless off-task requests.
- Every new row is checked against the frozen sets and against every other new set by normalised text: 0 overlaps
  with the frozen sets. The frozen files are byte-identical to before (SHA-256 `eef2da60…`, `e00b70c3…`).
  Hashes of all sets: `evidence/context_guard/datasets.json`.
- **Contamination flags.** `unseen_synth`'s injected strings come from the Engine 2 training candidate, so
  `roberta6l` and `scratch` were trained on them; their `unseen_synth` numbers are not a generalisation measure.
  deepset and fmops were trained on the frozen dev set's source.
- **Independence of `unseen_email`.** LLMail-Inject was published on 2025-05-02, a year after the pinned ProtectAI
  revision (2024-04-23). Neither LLMail nor Enron appears in the Engine 2 training candidate, and neither was used to
  fit the guard.
- `eval/context_sets/` is git-ignored like `eval/sets/` (it contains real Enron e-mails). It is rebuilt
  deterministically by the scripts in §9 and verified by its hashes. The committed feature caches contain no text.

## 5. Protocol

- The guard's fusion weights are fitted on **ctx_dev only**, with segment-level labels (the known injected
  segments).
- WARN and BLOCK thresholds for **every** model, baselines included, are chosen on ctx_dev with the same FPR-bounded
  rule as Engine 2 (BLOCK: precision ≥ 0.98, FPR ≤ 1%; WARN: recall ≥ 0.95, FPR ≤ 10%; documented fallbacks). They
  are then frozen. **No model reached its targets on ctx_dev** (all fell back), which is itself a finding: none of
  these detectors gives high recall at a low false-positive rate on embedded injections.
- Baselines score the same documents with the same production inference code (`ppaguard.engine2.Detector`), on the
  whole document as the app does today.
- Ablations reuse one cached feature matrix, so they differ only in the zeroed feature columns.
- Live/cached parity: the exported artifact, scored live, matches the cached evaluation path within 1.5e-6, and no
  document lies within 1e-5 of a threshold. The reported numbers therefore describe the shipped component.
- 95% bootstrap intervals for ROC-AUC, recall and FPR are in the full comparison.
- `hybrid_max` (the maximum of guard and ProtectAI) was defined **after** seeing the first unseen e-mail result. It is
  post hoc and labelled so everywhere. Its only clean test is the pre-registered confirmatory set.

## 6. Results

<!-- HEADLINE:BEGIN (generated by scripts/context_guard/report_context.py) -->
Every value below is generated from `docs/v5/evidence/context_guard/comparison.json` at the WARN threshold each model was given on ctx_dev. Full tables, BLOCK-band results, ablations and confidence intervals: [evidence/context_guard/comparison.md](evidence/context_guard/comparison.md).

**Overall** (ROC-AUC is threshold-free; P / R / F1 / FPR at WARN)

| Set | n (inj) | Context guard | ProtectAI (default) | hybrid_max (post hoc) | roberta6l | scratch | deepset | fmops |
|---|---|---|---|---|---|---|---|---|
| Frozen test | 1026 (600) | AUC 0.91<br>P 0.93 R 0.36<br>F1 0.52 FPR 0.04 | AUC 0.40<br>P 0.46 R 0.28<br>F1 0.35 FPR 0.47 | AUC 0.65<br>P 0.66 R 0.45<br>F1 0.54 FPR 0.34 | AUC 0.55<br>P 0.11 R 0.01<br>F1 0.02 FPR 0.10 | AUC 0.66<br>P 0.48 R 0.02<br>F1 0.04 FPR 0.03 | AUC 0.59<br>P 0.66 R 0.90<br>F1 0.76 FPR 0.67 | AUC 0.81<br>P 0.64 R 1.00<br>F1 0.78 FPR 0.78 |
| Unseen e-mail | 961 (398) | AUC 0.70<br>P 0.81 R 0.48<br>F1 0.60 FPR 0.08 | AUC 0.92<br>P 0.90 R 0.67<br>F1 0.77 FPR 0.06 | AUC 0.88<br>P 0.86 R 0.73<br>F1 0.79 FPR 0.08 | AUC 0.80<br>P 0.93 R 0.31<br>F1 0.47 FPR 0.02 | AUC 0.79<br>P 1.00 R 0.08<br>F1 0.15 FPR 0.00 | AUC 0.83<br>P 0.42 R 1.00<br>F1 0.59 FPR 0.97 | AUC 0.77<br>P 0.42 R 1.00<br>F1 0.59 FPR 0.98 |
| Confirmatory e-mail | 800 (400) | AUC 0.58<br>P 0.72 R 0.33<br>F1 0.45 FPR 0.12 | AUC 0.90<br>P 0.89 R 0.68<br>F1 0.77 FPR 0.08 | AUC 0.76<br>P 0.82 R 0.61<br>F1 0.70 FPR 0.13 | AUC 0.70<br>P 0.88 R 0.25<br>F1 0.39 FPR 0.03 | AUC 0.79<br>P 0.83 R 0.05<br>F1 0.09 FPR 0.01 | AUC 0.83<br>P 0.51 R 1.00<br>F1 0.67 FPR 0.96 | AUC 0.70<br>P 0.50 R 1.00<br>F1 0.67 FPR 0.99 |
| Unseen synthetic docs | 97 (35) | AUC 0.81<br>P 0.89 R 0.49<br>F1 0.63 FPR 0.03 | AUC 0.69<br>P 0.60 R 0.34<br>F1 0.44 FPR 0.13 | AUC 0.84<br>P 0.83 R 0.57<br>F1 0.68 FPR 0.06 | AUC 0.92<br>P 0.69 R 0.83<br>F1 0.75 FPR 0.21 | AUC 0.76<br>P 1.00 R 0.03<br>F1 0.06 FPR 0.00 | AUC 0.91<br>P 0.42 R 1.00<br>F1 0.59 FPR 0.77 | AUC 0.80<br>P 0.37 R 1.00<br>F1 0.54 FPR 0.95 |
| Frozen dev (direct; out of scope) | 659 (203) | AUC 0.88<br>P 0.93 R 0.06<br>F1 0.12 FPR 0.00 | AUC 0.82<br>P 0.75 R 0.47<br>F1 0.58 FPR 0.07 | AUC 0.89<br>P 0.80 R 0.42<br>F1 0.55 FPR 0.05 | AUC 0.72<br>P 0.73 R 0.12<br>F1 0.20 FPR 0.02 | AUC 0.66<br>P 0.75 R 0.03<br>F1 0.06 FPR 0.00 | AUC 0.97<br>P 0.73 R 1.00<br>F1 0.85 FPR 0.16 | AUC 0.98<br>P 0.72 R 1.00<br>F1 0.84 FPR 0.17 |

**Hidden-injection recall** (share of injected documents flagged, WARN)

| Subset | n | Context guard | ProtectAI (default) | hybrid_max (post hoc) | roberta6l | scratch | deepset | fmops |
|---|---|---|---|---|---|---|---|---|
| BIPIA hidden in e-mail | 150 | 0.33 | 0.73 | 0.67 | 0.00 | 0.00 | 1.00 | 1.00 |
| BIPIA hidden in table | 300 | 0.22 | 0.21 | 0.24 | 0.00 | 0.04 | 0.81 | 1.00 |
| BIPIA hidden in code | 150 | 0.67 | 0.00 | 0.67 | 0.03 | 0.00 | 1.00 | 1.00 |
| synthetic e-mail | 13 | 0.54 | 0.46 | 0.62 | 0.77 | 0.08 | 1.00 | 1.00 |
| synthetic document | 12 | 0.58 | 0.25 | 0.67 | 0.92 | 0.00 | 1.00 | 1.00 |
| synthetic table | 4 | 0.25 | 0.25 | 0.25 | 1.00 | 0.00 | 1.00 | 1.00 |
| synthetic code | 6 | 0.33 | 0.33 | 0.50 | 0.67 | 0.00 | 1.00 | 1.00 |
| LLMail phase-1 attack e-mail | 398 | 0.48 | 0.67 | 0.73 | 0.31 | 0.08 | 1.00 | 1.00 |
| LLMail phase-2 attack e-mail (confirmatory) | 400 | 0.33 | 0.68 | 0.61 | 0.25 | 0.05 | 1.00 | 1.00 |

**False-positive rate on benign content** (share flagged, WARN)

| Subset | n | Context guard | ProtectAI (default) | hybrid_max (post hoc) | roberta6l | scratch | deepset | fmops |
|---|---|---|---|---|---|---|---|---|
| NotInject trigger-word prompts (2) | 113 | 0.00 | 0.53 | 0.42 | 0.10 | 0.01 | 0.43 | 0.35 |
| NotInject trigger-word prompts (3) | 113 | 0.00 | 0.58 | 0.43 | 0.24 | 0.05 | 0.81 | 0.81 |
| NotInject trigger-word prompts (1, dev) | 113 | 0.00 | 0.25 | 0.16 | 0.04 | 0.02 | 0.65 | 0.69 |
| synthetic docs with trigger words | 27 | 0.04 | 0.26 | 0.11 | 0.33 | 0.00 | 0.96 | 1.00 |
| clean BIPIA e-mails | 50 | 0.08 | 0.74 | 0.64 | 0.00 | 0.00 | 1.00 | 1.00 |
| clean BIPIA tables | 100 | 0.00 | 0.39 | 0.03 | 0.00 | 0.07 | 0.43 | 1.00 |
| clean BIPIA code | 50 | 0.24 | 0.00 | 0.24 | 0.08 | 0.00 | 1.00 | 1.00 |
| LLMail benign FP-test e-mails | 160 | 0.03 | 0.00 | 0.03 | 0.00 | 0.00 | 1.00 | 1.00 |
| real Enron e-mails | 403 | 0.10 | 0.08 | 0.10 | 0.02 | 0.00 | 0.96 | 0.98 |
| real Enron e-mails (confirmatory) | 400 | 0.12 | 0.08 | 0.13 | 0.03 | 0.01 | 0.96 | 0.99 |
<!-- HEADLINE:END -->

## 7. Interpretation and trade-offs

<!-- INTERPRETATION:BEGIN -->
**Where the gain comes from.** The ablations isolate it. On the frozen test, dropping the off-topic feature
lowers ROC-AUC from 0.913 to 0.625, while dropping the localised detector changes little (0.902). BIPIA plants a
*task-irrelevant* instruction inside an otherwise coherent document, and a segment that is unrelated to its document
is exactly what the off-topic feature measures. The trigger-word false positives disappear for a different reason:
the guard needs an instruction to be off-topic, AI-addressed or structurally forged. A benign "should I ignore this
warning?" is none of these, while a whole-document classifier reacts to the word itself.

**Where it loses, and why.** On attack e-mails the same feature works against the guard. Dropping the off-topic
feature *raises* e-mail ROC-AUC (0.70 → 0.87 on unseen_email, 0.58 → 0.82 on the confirmatory set). The attack fills
the whole e-mail, so nothing in it is off-topic relative to the rest. Meanwhile real Enron mail has forwarded headers,
signatures and personal asides that *are* off-topic. Whole-document ProtectAI does better here because the
attacker's text is the whole input, which is the setting it was trained for.

**Trade-offs at the operating points.**

| Content | Guard vs the current default at WARN |
|---|---|
| Benign trigger-word prompts (NotInject, 339 rows) | 0% vs 25–58% flagged: large false-positive reduction |
| Clean BIPIA e-mails / tables | 8% / 0% vs 74% / 39% flagged |
| Clean code (BIPIA) | 24% vs 0% flagged: worse (mostly the delimiter cue matching code syntax, §8) |
| Real benign e-mail (Enron, 803 rows) | 10–12% vs 8% flagged: slightly worse |
| Hidden injections in BIPIA documents | recall e-mail 0.33 vs 0.73, table 0.22 vs 0.21, code 0.67 vs 0.00 |
| Attacker-written e-mails (LLMail, 798 rows) | recall 0.33–0.48 vs 0.67–0.68: worse |

Note on fmops and deepset: the shared rule picked 0.99 for both on ctx_dev, yet at that threshold they still flag
67–99% of benign rows on every set except frozen dev, where the FPR is 0.16–0.17 (that set comes from their training
source). Their outputs saturate near 1.0, so their ROC-AUC is the fairer comparison.

**Recommendation.** Keep the current production configuration: ProtectAI as Engine 2, Engine 1 isolation, the rule
signals and analyst review. Do not switch to the guard. The justified next step is a **document-aware second stage**:
a small model fitted on development data that contains *both* regimes (embedded injections and attacker-written
messages, plus standalone benign trigger-word text). It would combine the guard's score, the whole-document score
and the document's structure (segment count, tables, code), and be tested once on a fresh, pre-registered set. A
naive maximum does not work (H2 failed), because the two scores live on different scales and thresholds. §10 reports
that experiment.
<!-- INTERPRETATION:END -->

## 8. Limitations

1. **Small development set.** ctx_dev has 114 documents. Thresholds selected on it are noisy, and the bootstrap
   intervals on the smaller unseen sets are wide (see `comparison.md`).
2. **Researcher bias.** The off-topic feature and the "task-irrelevant" dev category were designed with published
   knowledge of how BIPIA builds its attacks (task-irrelevant instructions appended to documents). No test data was
   used for fitting or thresholds, but the large frozen-test gain should be read as *this feature works for
   task-irrelevant embedded instructions*, not as general robustness. The independent e-mail sets are the stronger
   evidence.
3. **Standalone malicious content.** An e-mail whose whole body is the attack gives the off-topic feature nothing to
   contrast against (predicted in [THREAT_MODEL.md §6](THREAT_MODEL.md)). This is where the guard is weakest.
4. **Clean code and real e-mail.** At WARN the guard flags 12 of the 50 clean BIPIA code files. Measured per flagged
   file (the feature with the largest positive contribution in the top segment), the main driver is the
   **delimiter-forgery cue** (5 of 12), which matches ordinary code syntax (`//`, `}}`, `==`). Off-topic (3),
   second person (2) and the localised detector (2) account for the rest. On real Enron e-mails, 40 of 403 are flagged,
   mainly through off-topic segments such as forwarded headers and signatures (18), directive verbs in requests to
   people (9) and second person (7). Excluding delimiter cues inside code segments is the obvious fix. It was found by
   looking at test data, so it is listed as a next step and **not** applied to these results.
5. **Direct prompts are out of scope.** On the frozen dev set's deepset prompts (a user typing an injection), recall
   is very low. The guard is meant for the data channel, and direct prompts stay with Engine 2.
6. **English only.** Lexical cues and MiniLM are English-centric.
7. **Not adversarially evaluated.** The features are public. An attacker who writes on-topic, cue-free text, or
   splits a payload across segments, can evade them. The guard raises the cost of the templated attacks in these
   datasets; it is not a robust defence.
8. **Recall at low FPR remains modest everywhere** (§6). The guard is a filter for human review, not a gate.

## 9. Reproduce

Needs the frozen sets (`eval/sets/`) and the audited training candidate
(`experiments/results/injecguard_audit_v2/candidate_train.jsonl`, the TaskTracker source), as for Engine 2 training.

```bash
python scripts/context_guard/download_sources.py        # LLMail-Inject @1063bdf0 + Enron byte ranges, ~520 MB, ~1 min
python scripts/context_guard/build_datasets.py          # ctx_dev, unseen_synth, unseen_email (+ manifest)
python scripts/context_guard/build_confirm_set.py       # unseen_email_2 (confirmatory)
# features: the only heavy step (~0.2-1 s per document on 2 CPU cores); resumable; one set per call
for s in ctx_dev frozen_dev frozen_test unseen_synth unseen_email unseen_email_2; do
  python scripts/context_guard/cache_features.py --set $s
  python scripts/context_guard/add_baseline_scores.py --set $s --model deepset=<dir> --model fmops=<dir>
done
python scripts/context_guard/evaluate_from_cache.py     # fit + thresholds on ctx_dev, evaluate all (seconds)
python scripts/context_guard/report_context.py          # comparison.md + the tables in this file
python scripts/context_guard/export_guard_artifact.py   # artifact + live/cached parity check
pytest tests/v5/test_v5_context_guard.py
```

- **Verified:** a fresh `download_sources.py` followed by both builders reproduced all four new sets **byte for byte**
  (the SHA-256 values in `evidence/context_guard/datasets.json`).
- Model folders: the detector defaults to the app's own ProtectAI folder (`$PPAGUARD_CACHE/protectai-v2`, prepared
  through the Engine 2 registry). MiniLM is downloaded on first use to `$PPAGUARD_CACHE/minilm`
  (`sentence-transformers/all-MiniLM-L6-v2` @`1110a243`). The deepset and fmops folders come from
  `scripts/prepare_hf_classifier.py` (see EVALUATION.md). Override any of them with `--detector` / `--minilm` /
  `--model`.
- The committed feature caches (`evidence/context_guard/cache/`, no text) let `evaluate_from_cache.py` and
  `report_context.py` be re-run with **no models and no datasets**.

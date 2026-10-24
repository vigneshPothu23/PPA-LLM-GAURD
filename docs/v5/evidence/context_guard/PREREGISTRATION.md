# Pre-registration: confirmatory e-mail test (written before the confirmatory set was scored)

Written 2026-10-24T07:43Z, before any model was run on `eval/context_sets/unseen_email_2.jsonl`
(SHA-256 `40dd399fefc672c02bbdd687c7c13063e240a4fd2707e6d473ca131054bddb41`, 400 LLMail-Inject phase-2
attack e-mails + 400 fresh Enron e-mails, disjoint from every earlier set). Committed to git before scoring,
so the commit timestamp precedes the results commit.

## What motivated it

On the first unseen e-mail set (`unseen_email`), the full context guard was **worse** than the whole-document
ProtectAI detector (ROC-AUC 0.701 vs 0.918; attack recall at WARN 0.48 vs 0.67). The threat model (§6.1)
predicted this: an e-mail whose whole body is the attack has no benign context for the off-topic feature to
contrast against. Any fix designed after seeing that result must be tested on new data.

## Fixed before scoring

- **Models** (no refitting): `guard_full`, `guard_no_offtopic`, `protectai_wholedoc`, and
  `hybrid_max` = max(`guard_full`, `protectai_wholedoc`) per document. `hybrid_max` was defined after the first
  unseen e-mail result; it is post hoc and labelled so everywhere.
- **Thresholds:** WARN and BLOCK for every model, including `hybrid_max`, selected on `ctx_dev` with the
  FPR-bounded rule (as for every other model). No threshold is chosen on any test or unseen set.
- **Hypotheses:**
  - **H1.** ROC-AUC on `unseen_email_2`: `protectai_wholedoc` > `guard_full`.
  - **H2.** At the WARN threshold, `hybrid_max` attack recall ≥ `protectai_wholedoc` attack recall.
  - **H3 (the cost).** At the WARN threshold, `hybrid_max` benign FPR ≥ `guard_full` benign FPR.
- Results are reported whether or not they support the hypotheses.

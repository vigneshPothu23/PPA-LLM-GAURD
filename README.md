# PPA Guard: prompt-injection defence for an organization's LLM applications

PPA Guard screens untrusted text (user messages, e-mails, documents, code) before an LLM application uses it:

```
sign-in -> role + organization + application scope
  -> Engine 2: transformer detector (ONNX, CPU) -> calibrated P(injection)
  -> Policy: ALLOW / WARN / BLOCK (model-card thresholds + rule signals; fail closed)
  -> Engine 1: polymorphic prompt assembly (fresh random boundary per request), only when forwarding
  -> audit record -> result (your application calls its own LLM with the protected prompt)
```

**v5** (`src/ppaguard/`) is the current application. It is written independently of the earlier implementation:

| Part | Where | What |
|---|---|---|
| Engine 1 | `src/ppaguard/engine1/` | Per-request random boundaries (8 families × 8 tags × 48-bit nonce), 3 instruction templates, collision refusal, boundary-forgery signals, response marker check |
| Engine 2 | `src/ppaguard/engine2/` | Data validation, leakage guard, tokenizer verification, fine-tuning or from-scratch training, ONNX export + int8 quantization, temperature calibration, threshold selection on the frozen dev set, one-shot test evaluation, windowed inference, occlusion explanations |
| Policy | `src/ppaguard/policy/` | ALLOW / WARN / BLOCK, industry profiles, never releases a BLOCK, re-check on release |
| Core | `src/ppaguard/core/` | Five roles enforced on the server, organization and application scope, hash-chained audit, OIDC sign-in, security workflow |
| Integrations | `streamlit_app.py`, `src/ppaguard/sdk.py` | Streamlit console (teal + charcoal) and Python SDK (`PPAGuardClient`) |

**Engine 2 status: experimental.** Six detectors were compared on the frozen dev and test sets: ProtectAI, deepset
DeBERTa, a DistilBERT detector, and our own RoBERTa, from-scratch and fp32 variants. None is validated for production;
hidden injections inside e-mails and documents and harmless trigger-word prompts are the main failures. The default
(ProtectAI v2, pinned) is labelled EXPERIMENTAL throughout the app. See [EVALUATION.md](docs/v5/EVALUATION.md).

**Experimental: context guard** (`src/ppaguard/experimental/context_guard/`, not wired into the app). This is a
context-aware detector for injections hidden inside e-mails, documents, tables and code. It scores segments instead
of whole documents and combines lexical cues, the Engine 2 detector run per segment, and an off-topic signal; the
result feeds the unchanged ALLOW/WARN/BLOCK policy. It was evaluated against all six existing detectors on
identical inputs: the frozen sets, fresh LLMail-Inject and Enron e-mail sets, and synthetic and embedded-in-e-mail
document sets, with two pre-registered confirmatory tests. It beats the default on instructions hidden inside
documents, but it is worse on attacker-written e-mails and on real e-mail that contains trigger words. It is
**not** validated for production, and the default is unchanged. See [CONTEXT_GUARD.md](docs/v5/CONTEXT_GUARD.md) and the
[threat model](docs/v5/THREAT_MODEL.md).

Documentation: [Architecture & security](docs/v5/ARCHITECTURE.md) · [Engine 1](docs/v5/ENGINE1.md) ·
[Engine 2](docs/v5/ENGINE2.md) · [Evaluation](docs/v5/EVALUATION.md) · [Setup, training, deployment](docs/v5/DEPLOY.md) · [Build report](docs/v5/REPORT.md)

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PPAGUARD_DEMO_MODE = "1" ; streamlit run streamlit_app.py
```

## Preserved earlier work (unchanged)

- `src/ppa_guard/`: the previous engines (v3) and the v4 platform layer.
- `legacy/`: the v4 console and the single-page demo.
- `app_pages/`: v4 console pages.
- `eval/`, `experiments/`: evaluation and dataset-audit scripts and results.
- `docs/` (outside `docs/v5/`): earlier reports.

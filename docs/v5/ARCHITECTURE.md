# PPA Guard v5: architecture and security

A single Streamlit application with one Python package (`src/ppaguard/`) and one database. There are no microservices,
agents, vector stores or message queues. It is designed for an IT company of about 200 people, whose coding, e-mail and
documentation assistants pass untrusted text to an LLM. PPA Guard screens that text, decides **ALLOW / WARN / BLOCK**,
wraps permitted text with Engine 1, and records every decision.

```
src/ppaguard/
  engine1/   polymorphic prompt assembly, boundary-forgery signals            (no ML; stdlib only)
  engine2/   data, tokenizer, training, evaluation, calibration, ONNX runtime  (training needs torch; runtime does not)
  policy/    ALLOW/WARN/BLOCK decision engine, industry profiles, redaction
  core/      roles + authorization, database, audit chain, services, identity, security workflow, model loader
  sdk.py     Python integration interface (PPAGuardClient)
  ui/        Streamlit pages (teal + charcoal)
streamlit_app.py   entry point (Streamlit Community Cloud)
artifacts/engine2/   ProtectAI evaluation record + our trained experimental models (ONNX + tokenizer + model card)
```

The earlier implementation (`src/ppa_guard/`, `app_pages/`, `legacy/`) is kept unchanged for comparison. Nothing in
`src/ppaguard`, the UI or the example imports it. A test enforces this, and another test runs the whole workflow with the
old package blocked from import.

## Request workflow (`core/workflow.py`, `SecurityWorkflow.submit`)

| # | Step | Component | On failure |
|---|---|---|---|
| 1 | Identity: OIDC verified e-mail (or explicit demo persona) → invited, active member; role and organization re-read from the database on every request | `core/identity.py`, `core/authz.py` | no access |
| 2 | Authorization: role permission, organization, application scope (assigned/owned), application active | `core/authz.py` | Forbidden / NotFound |
| 3 | Rate limit per caller per application | workflow | RateLimited (audited) |
| 4 | Input validation (type, empty, length, NUL) | `policy/engine.py` | BLOCK |
| 5 | **Engine 2** threat analysis: calibrated P(injection), overlapping windows for long text, optional occlusion evidence | `engine2/runtime.py` | BLOCK (fail closed) |
| 6 | **Engine 1** boundary-forgery scan (signal) | `engine1/forgery.py` | — |
| 7 | **Policy**: thresholds from the model card (or a labelled override) + rule signals → most severe wins | `policy/engine.py` | — |
| 8 | WARN: hold for analyst review, or forward with a flag (application setting). A held WARN may consume a single-use analyst release, but only after this full re-check, and never for a BLOCK | workflow | — |
| 9 | **Engine 1** assembly, only when forwarding | `engine1/assembler.py` | BLOCK |
| 10 | One transaction: scan record + review + optional redacted text + audit event | workflow | BLOCK, nothing returned |
| 11 | Result: Engine 2 findings, policy decision, Engine 1 output, plain-language action | UI / SDK | — |

## Roles (backend-enforced; matrix in `core/roles.py` and on the System page)

| Role | Can do |
|---|---|
| Platform Administrator | Manage members and roles, all applications and policies, API keys, reviews, audit log and verification, view the model card |
| Security Analyst | Scan any application, view all scans (and retained redacted text), decide WARN reviews, view the audit log and model card |
| Application Developer / Integrator | Register applications; manage policy, members and API keys of the applications they own; scan their apps; view the model card |
| Auditor | Read-only: scans (metadata), applications, members, audit log and verification, model card; cannot scan or change anything |
| Member | Scan assigned applications; see only their own scans |

## Security properties and where they are tested

| Property | Tests |
|---|---|
| Authentication: unknown, uninvited or disabled identities refused; unverified Google e-mail refused; Microsoft only from allowed tenants; no sign-in configured → no data | `test_v5_core.py`, `test_identity.py` (shared identity logic), `test_v5_ui.py` |
| Authorization on every action, per role | `test_v5_core.py` (parametrised role tests), `test_v5_ui.py` (pages per role) |
| Organization isolation (NotFound across orgs; composite foreign keys) | `test_v5_core.py` |
| Application scope, paused applications (console and API key) | `test_v5_core.py`, `test_v5_sdk_and_independence.py` |
| ALLOW / WARN / BLOCK, rule floors, strict thresholds, never release a BLOCK, re-check on release, single use, expiry, concurrency | `test_v5_policy.py`, `test_v5_core.py`, `test_v5_ui.py` |
| Fail closed: detector missing or erroring, model files missing or corrupted, record failure, invalid input, Engine 1 refusal | `test_v5_core.py`, `test_v5_engine2.py`, `test_v5_policy.py`, `test_v5_sdk_and_independence.py` |
| Data minimisation: no raw text stored; redacted retention per profile; evidence not stored; separator never stored | `test_v5_core.py` |
| Audit chain: tamper, deletion and truncation detected; concurrent appends | `test_v5_core.py` |
| Python integration through the real workflow; key scope and revocation | `test_v5_sdk_and_independence.py` |

## Integrations actually supported

1. **Streamlit console**: everything above.
2. **Python SDK**: `PPAGuardClient(api_key).protect(text)` from an approved Python application that shares the
   console's database. A runnable example is in `examples/integration_example.py`.

PPA Guard is not a proxy or an HTTP API. It does not intercept traffic from applications that do not call it, and it
does not call an LLM.

## Knowledge Base: not built

A knowledge base would need retrieval and, for answers, an LLM. That means a paid API or a model too large for the free
tier, and it would open a new path for untrusted text to reach an LLM. None of the required workflows need it. The model
card page, the System page and these docs cover the reference material.

## Known limitations

- Without `PPAGUARD_DATABASE_URL`, Streamlit Community Cloud uses a local SQLite file, which is lost whenever the app
  restarts.
- The audit chain shows tampering after the fact. It cannot stop someone with direct database write access from
  rewriting the whole chain.
- The rate limit counts recorded scans, so concurrent bursts can slightly exceed it.
- Demo mode is not authentication. It uses its own database and shows a banner on every page.
- Engine 1's effect on attack success was not measured, because no LLM is connected.

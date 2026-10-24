# Build report: PPA Guard console (2026-10-22)

## Verification stages

| Stage | Status | Evidence |
|---|---|---|
| **1. Build** | Done | A clean Python 3.13 virtual environment was created with only `requirements.txt` and pytest. `pip check` found no broken requirements. 283 passed, 3 skipped (`docs/evidence/test_run.txt`). |
| **2. Local run** | Done, with the limits below | The app ran with `streamlit run` and was driven in Chromium through Playwright. all 18 checks passed (`docs/evidence/screenshots/walkthrough_log.txt`; the unconfigured-mode check appears twice because it was re-run on the final build), with 12 screenshots. It also ran with a random-weight model of the official architecture (738 MB ONNX). The full walkthrough passed and peak RSS was **1.38 GB**. |
| **3. Public deployment** | **Not done** | The build sandbox has no GitHub or Streamlit account, and outbound access to Hugging Face, Google and Microsoft is blocked. Follow `docs/DEPLOY.md` §2, then run the §3 checks against your URL. |

Limits of stage 2:
- The real ProtectAI checkpoint was not available here, so every score in the screenshots comes from a stub model with
  random weights, and the UI says so.
- OIDC was verified up to the redirect start only. Fetching the provider metadata was blocked by the sandbox proxy.
  Claim handling, meaning the verified-email and Microsoft-tenant rules, is covered by unit tests.

## Requirements, and how each was met

| Requirement | Result |
|---|---|
| Engine 1 preserved | Every file under `src/`, `eval/` and `tests/fixtures` is byte-identical to the 2026-10-18 baseline (`cmp`). The 73 SDK-equivalence tests pass. |
| Engine 2 (pretrained DeBERTa via ONNX) | Unchanged. It is loaded through `platform/runtime.py`, which reports what it actually loaded: pinned download, unverified local folder, or failure. It never substitutes another model or a canned score. |
| ALLOW / WARN / BLOCK | `policy.py` is unchanged. Each application takes its thresholds and rule actions from an industry profile (General, Software/IT, Healthcare, Finance), which can be edited within bounds. Any threshold other than 0.50 / 0.92 is labelled "not validated". A WARN is either held for review or forwarded with a flag. |
| RBAC: five roles, enforced on the backend | `platform/roles.py` and `authz.py`. Each role is tested on every privileged action, on SQLite **and** PostgreSQL 16. |
| Integrations | Two are supported: the console, and the in-process Python SDK with a revocable API key bound to one application. There is no proxy and no claim of universal gateway protection (System & Integrations page, `ARCHITECTURE_SECURITY.md`). |
| Knowledge Base | Not built. The reason is in `ARCHITECTURE_SECURITY.md`. |
| Teal / charcoal UI | `.streamlit/config.toml` and `app_pages/ui.py`, using the agreed palette (#0C1113 / #171D20 / #1FB5A3), Inter and JetBrains Mono. |
| Streamlit Community Cloud, zero cost | Feasible on paper: about 1.38 GB proxy peak against the documented 2.7 GB cap. Not deployed. See `DEPLOY.md`. |
| No microservices, agents, vector DB or complex tenancy | One process, one database. Organizations are rows. |

## Test evidence (`docs/evidence/`)

| Suite | Tests | What it proves |
|---|---|---|
| `tests/test_platform.py` | 49 × 2 backends (SQLite + PostgreSQL 16) − 1 | The five rules, every role boundary, cross-organization `NotFound`, the composite-foreign-key backstop, ALLOW/WARN/BLOCK, single-use release (including 8 concurrent attempts), release expiry, BLOCK never released, fail-closed (detector missing or erroring, record failure), rate limit, retention and redaction, raw text never stored, API key scope and revocation, audit tamper and truncation detection, concurrent audit appends, and the real ONNX code path with the stub model |
| `tests/test_identity.py` | 14 | Mode selection fails closed; verified-email rules; Microsoft tenant allow-list; first-admin bootstrap is idempotent; model loader reports failure, unverified folders and checksum mismatch |
| `tests/test_app.py` | 10 | Headless UI: unconfigured app shows nothing; each of the five roles gets exactly its pages and every page renders; a member scan creates a record; a missing model blocks; the full WARN → analyst release → single forward → held-again cycle |
| `tests/test_legacy_demo.py` | 2 | The previous single-page demo, kept in `legacy/`, still runs |
| Existing suites | 163 | Engine 1 SDK equivalence (73) and unit tests (12), fusion (21), the experiments suites (54: audit v1 23, audit v2 13, probe 18). The 3 real-model tests are skipped because no checkpoint is present. |

## File-level change summary

**New**
- `src/ppa_guard/platform/`: `roles.py`, `db.py`, `policies.py`, `audit.py`, `authz.py`, `gateway.py`, `services.py`, `identity.py`, `runtime.py`, `__init__.py`
- `app_pages/`: `ui.py`, `command_centre.py`, `scan_console.py`, `review_queue.py`, `history.py`, `applications.py`, `members.py`, `audit_log.py`, `system.py`, `__init__.py`
- `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `assets/logo.svg`
- `tests/test_platform.py`, `tests/test_identity.py`, `tests/test_legacy_demo.py`
- `legacy/single_page_demo.py`: the previous `streamlit_app.py`, preserved. Only its `ROOT` path changed, so it still runs from `legacy/`.
- `docs/ARCHITECTURE_SECURITY.md`, `docs/DEPLOY.md`, `docs/BUILD_REPORT.md`, `docs/evidence/`

**Changed**
- `streamlit_app.py`: rewritten as the multi-page, role-aware console entry point.
- `tests/test_app.py`: rewritten for the new console. The old tests moved to `tests/test_legacy_demo.py`.
- `requirements.txt`: adds `sqlalchemy==2.0.55`, `psycopg[binary]==3.3.6` and `Authlib==1.9.1`.
- `pyproject.toml`: version 4.0.0 and a new description.
- `.gitignore`: ignores local databases.
- `README.md`, `docs/STATUS.md`.

**Unchanged, verified by hash:** `experiments/scripts/audit_injecguard.py` (`9dd27891…296e`), `audit_injecguard_v2.py`
(`5e45c03e…1944b`), `experiments/tests/test_audit_injecguard.py` (`4d6b9807…bd5d4`), `test_audit_injecguard_v2.py`
(`7df4792d…2b55a47`) and `eval/compare.py` (`9ae12bde…501d`). No file in `experiments/results/` was modified. Nothing
was deleted.

## Data-integrity tasks from the master prompt

| Task | Status |
|---|---|
| Local validation/test hashes differ | **Likely cause: Windows line endings.** The sandbox copies match the expected LF hashes exactly: validation `eef2da60…5bc8` (659 lines, 121,715 bytes), test `e00b70c3…9266` (1,026 lines, 1,119,899 bytes). `build_eval_sets.py` writes in text mode, so on Windows it produces CRLF files. Their hashes would be `ef2270c0a8719bdd8afc7daaefbed2447278b65e68f6302e60e72c1bba365ec7` (validation, 122,374 bytes) and `08525d6edb6fad523f118aa2260e9c2412c4025dade88b7c4a924aacb0463c5b` (test, 1,120,925 bytes). If your hashes and sizes match these, the content is identical and only the line endings differ. Nothing needs replacing. |
| Rebuild the v2 audit outputs | Must run on your machine; the recovery script delivered earlier does it. **Not run here** against your data. |
| Tokenizer verification | **Blocked here**: Hugging Face is unreachable from the sandbox. |
| Training | **Not started.** The readiness gate has not passed on your data. |

PowerShell check that reads your files and changes nothing:

```powershell
foreach ($f in "eval\sets\validation.jsonl","eval\sets\test.jsonl") {
  $b = [IO.File]::ReadAllBytes($f); $crlf = ([Text.Encoding]::UTF8.GetString($b) -split "`r`n").Count - 1
  $lf = [Text.Encoding]::UTF8.GetBytes([Text.Encoding]::UTF8.GetString($b).Replace("`r`n","`n"))
  $h = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($lf)).Replace("-","").ToLower()
  "$f bytes=$($b.Length) CRLF=$crlf LF-normalised-sha256=$h"
}
```

## Blockers

1. **Public deployment** needs your GitHub repository, a Streamlit Community Cloud login, a Neon project and OAuth
   clients. These are manual steps (`DEPLOY.md` §2).
2. **Real-model behaviour and memory** cannot be checked in the sandbox (Hugging Face is blocked). Run
   `tests/test_real_model.py` and `eval/compare.py` locally, and check memory in the Cloud logs.
3. **OIDC end-to-end login** needs real client IDs and network access to Google or Microsoft. Verify after deployment.
4. **Community Cloud resource figures** come from documentation dated February 2024 that "may change". Fit is likely
   but unproven until you deploy.

# PPA Guard: architecture and security

PPA Guard is one Streamlit application. It runs in one process, uses one database, and has no services, queues, agents
or vector store. It screens untrusted text before an LLM application uses it. It returns a PPA-protected prompt only
when the policy allows it, and it records every decision.

## Components

| Part | Code | Notes |
|---|---|---|
| Engine 1 | `src/ppa_guard/engine1/` | Polymorphic Prompt Assembling. Byte-identical to the authors' SDK (commit b3c20a3): 84 separators, 72 distinct, tested in `tests/test_engine1_equivalence.py`. The app always uses hardened mode, which never picks a separator that already appears in the input. **Unchanged.** |
| Engine 2 | `src/ppa_guard/engine2/` | `protectai/deberta-v3-base-prompt-injection-v2`, pinned revision `89b085cd…`, official ONNX export, CPU only. Pretrained and used as is. **Unchanged.** |
| Policy | `src/ppa_guard/policy.py` (unchanged) and `platform/policies.py` | The fusion rules are unchanged. `policies.py` adds four industry profiles and per-application settings, each validated within bounds. |
| Gateway | `platform/gateway.py` | The only path that produces a protected prompt. |
| Authorization | `platform/roles.py`, `platform/authz.py` | Five roles and a permission table. Every check runs on the server. |
| Services | `platform/services.py` | Members, applications, policies, API keys, reviews and audit. Each function authorizes its own caller. |
| Audit | `platform/audit.py` | A hash chain kept per organization. |
| Identity | `platform/identity.py`, `streamlit_app.py` | Google or Microsoft OIDC through `st.login`, or an explicit demo mode. |
| Storage | `platform/db.py` | SQLAlchemy Core. SQLite locally; PostgreSQL in production (for example Neon Free). |
| UI | `streamlit_app.py`, `app_pages/` | Teal and charcoal theme. `st.navigation` shows each role only its own pages. |
| LLM | `src/ppa_guard/llm.py` | A placeholder. No LLM is ever called. |

## Request path (`Gateway.submit`)

```
caller (signed-in user, or API key bound to one application)
  1  authorize: role permission + organization + application scope     -> Forbidden / NotFound
  2  rate limit per user (or key) per application                     -> RateLimited (audited)
  3  Engine 2 detection (outside the transaction)                      -> error => BLOCK (fail closed)
  4  policy: rule floor + score band, application profile              -> ALLOW / WARN / BLOCK
  5  WARN: hold for review; or forward with a flag if the app allows it; or consume a single-use release
  6  Engine 1 assembles the protected prompt (only if forwarding)       -> separator collision => BLOCK
  7  one transaction: scan record + review + optional redacted text + audit event
                                                                        -> any failure => BLOCK, nothing returned
  8  return decision, reasons, evidence, protected prompt (if forwarded); LLM placeholder note
```

A BLOCK never reaches Engine 1. A held WARN never reaches Engine 1. A protected prompt is returned only after its
record has committed.

## Roles (enforced in `roles.py` and `authz.py`; the matrix is also shown in the app)

| Role | Can do |
|---|---|
| Platform Administrator | Everything in the organization: members, roles, every application and policy, reviews, the audit log and its verification. |
| Security Analyst | Scan any application. View every scan, including retained redacted text. Decide reviews. View the audit log. Cannot change members or policies. |
| Application Developer / Integrator | Register applications, then manage policy, members and API keys for the applications they own. Scan applications they own or are assigned to. See scans in their own applications. |
| Auditor | Read-only access to scans (metadata only), applications, members and the audit log. Can verify the chain. Cannot scan or change anything. |
| Member | Scan the applications they are assigned to. See only their own scans. |

## The five rules, and where each is enforced and tested

| Rule | Enforcement | Tests |
|---|---|---|
| 1. Authentication | Google/Microsoft OIDC. The identity must be a verified e-mail address: Google requires `email_verified`; Microsoft is accepted only from tenants listed in `microsoft_tenant_ids`. The address must belong to an invited, active member. Membership is read again from the database on every rerun. With no sign-in configured, the app shows nothing. There is no password login. | `test_identity.py`, `test_unknown_or_missing_identity_is_refused`, `test_disabled_user_loses_access_on_next_request`, `test_unconfigured_sign_in_fails_closed`, browser walkthrough |
| 2. Authorization on every action | Each service and gateway function calls `require()` or `app_for()`. The UI hides pages only as a convenience. | Parametrized tests per role in `test_platform.py`; `test_each_role_sees_only_its_pages_and_every_page_renders` |
| 3. Organization isolation | The organization comes from the caller's membership. Every query filters on `org_id`. Objects in another organization raise `NotFound`. Composite `(org_id, id)` foreign keys reject cross-organization rows in the database. Disabling a member applies to one organization only. | `test_other_org_objects_are_not_found`, `test_composite_foreign_key_blocks_cross_org_rows`, `test_disabling_in_one_org_does_not_touch_another_org` |
| 4. Application scope | Members and developers can scan only applications they are assigned to or own. Developers manage only applications they own. An API key works for one application and only for scanning. A paused application refuses scans. | `test_member_scans_only_assigned_apps`, `test_developer_manages_only_own_application`, `test_api_key_scoped_to_one_app_and_revocable`, `test_paused_application_refuses_scans` |
| 5. Data minimisation | Each scan stores a SHA-256 of the input, its length and the decision metadata. Raw text is never stored. Redacted text is kept only when the application enables retention: up to 30 days on general and software, 7 on finance, never on healthcare. It is deleted at expiry and never shown to Auditors. Evidence sentences are returned to the caller but not stored. | `test_raw_prompt_is_never_stored_by_default`, `test_retention_stores_redacted_text_only_and_expires`, `test_evidence_returned_but_not_stored` |

## WARN review and single-use release

A held WARN creates a review. Only a Security Analyst or a Platform Administrator can decide it, and nobody can decide
their own review. A release is bound to the requester, the application and the SHA-256 of the exact input. It expires
after 24 hours and can be used once. The same text is still scanned in full when it is resubmitted. A release never
overrides a BLOCK. The conditional `UPDATE ... WHERE status='released'` makes the release single-use even under
concurrency. This is tested with 8 parallel submissions on both SQLite and PostgreSQL.

## Audit chain

For each organization, `hash = SHA-256(prev_hash + canonical event)`. `verify` reports the first broken sequence
number. It detects edited, deleted, re-ordered and truncated-tail events (tested). Appends are serialised by a process
lock plus `SELECT … FOR UPDATE` on PostgreSQL, or `BEGIN IMMEDIATE` on SQLite. **Limitation:** someone with direct write
access to the database can rewrite the whole chain. Export the log regularly to keep an outside copy.

## Fail-closed cases (all tested)

- No sign-in configured: no data is shown.
- Unknown, disabled or uninvited identity: no access.
- Engine 2 missing or raising an error: BLOCK.
- Empty input or input over the limit: BLOCK.
- PPA separator collision: BLOCK.
- The decision record cannot be written: BLOCK, with no prompt returned.

## Supported integrations (and what is not supported)

1. **Console:** people use this app.
2. **In-process Python SDK:** an approved Python application imports `ppa_guard` and calls
   `Gateway.submit_with_api_key()`, against the same database. The full key is shown once, only its hash is stored,
   and it can be revoked.

PPA Guard is **not** a network proxy, an HTTP API or a browser extension. It cannot see or stop traffic from
applications that do not call it, and it does not call an LLM. It protects only inputs submitted to it, and only as
well as the pretrained detector and PPA work on those inputs. No accuracy figures are claimed for the real checkpoint in
this repository's app.

## Knowledge Base: not built

A knowledge base or assistant would add a vector store or an LLM. That would mean a paid API, or a model too large for
the free tier. It would also give untrusted text a new path to an LLM, which this project exists to guard. It is not
needed for ALLOW, WARN or BLOCK, for RBAC, or for the integrations. The **System & Integrations** page carries the
reference material an assistant would otherwise answer: component status, the integration boundary and the role matrix.

## Known limitations

- Streamlit Community Cloud runs one container. Without `PPA_DATABASE_URL`, SQLite data is lost on every restart.
- The rate limit counts recorded scans, so a burst of concurrent requests can slightly exceed it.
- In demo mode, choosing a persona is not authentication. Demo mode uses its own local database and always shows a banner.
- OIDC login against real Google or Microsoft servers was not completed in the build sandbox, because outbound access
  to those servers was blocked. The redirect start was verified, and claim handling is unit-tested.

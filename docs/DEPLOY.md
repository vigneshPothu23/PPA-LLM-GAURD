# Setup and deployment (zero cost)

## 1. Run locally with demo users (no accounts needed)

```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PPA_DEMO_MODE = "1"
streamlit run streamlit_app.py
```

- On first start, about 740 MB of ONNX files download from Hugging Face. This is free and needs no account. To use a
  folder you already downloaded, set `$env:PPA_MODEL_DIR = "C:\path\containing\onnx-parent"`.
- Demo mode seeds two synthetic organizations, "Acme Software (demo)" and "City Clinic (demo)", with one persona per
  role. Data is stored in `ppa_guard_demo.db`, a separate local file.
- Tests: `pip install -r requirements-dev.txt ; pytest`. To also test against PostgreSQL, set `PPA_TEST_POSTGRES_URL`.

## 2. Production on Streamlit Community Cloud

| Need | Free option | Notes |
|---|---|---|
| Hosting | Streamlit Community Cloud | Documented limits are 0.69–2.7 GB RAM and up to 2 CPU cores; the docs say these values date from Feb 2024 and may change. Apps sleep after 12 h without traffic. |
| Database | Neon Free (PostgreSQL) | 1 GB storage per project, 100 CU-hours per project per month, scales to zero after 5 min of inactivity (Neon free-plan FAQ, checked 2026-10-22). |
| Sign-in | Google OAuth client (Google Cloud console) and/or Microsoft Entra app registration | Both are free for sign-in. |
| Model | Hugging Face public download | Free; no token needed for this public model. |

Steps:

1. **Push the repository** to GitHub. Public or private both work on Community Cloud. `.gitignore` already excludes
   models, evaluation data, local databases and `secrets.toml`.
2. **Create the database.** In Neon, create a project and copy its connection string (`postgresql://…?sslmode=require`).
3. **Create the sign-in client(s).** The redirect URI is `https://<your-app>.streamlit.app/oauth2callback`.
   - Google: Google Cloud console → APIs & Services → Credentials → OAuth client ID (Web application). Add the redirect URI.
   - Microsoft: Entra ID → App registrations → New registration (Web). Add the redirect URI and create a client secret.
     Note the **Directory (tenant) ID**: Microsoft identities are accepted only from tenants listed in
     `microsoft_tenant_ids`.
4. **Deploy:** share.streamlit.io → Create app → your repository, branch, main file `streamlit_app.py`, Python 3.12.
5. **Secrets:** paste the filled-in contents of `.streamlit/secrets.toml.example` into the app's Secrets box. These
   include `PPA_DATABASE_URL`, `[auth]`, the providers, `cookie_secret`, and `[ppa] bootstrap_admin_email` (your address).
   Do **not** set `demo_mode` in production.
6. **First sign-in:** when the database is empty, the app creates your organization and makes the bootstrap address
   its Platform Administrator. Invite everyone else from **Members**. Only invited addresses get access.

## 3. Verify the deployment (do this before relying on it)

These checks run against the real deployment. They have **not** been run by the build, because it had no GitHub or
Streamlit account and outbound access to Hugging Face, Google and Microsoft was blocked.

1. **System & Integrations:** Engine 2 shows `protectai/deberta-v3-base-prompt-injection-v2@89b085cd330414…` with a
   green dot. Database shows `postgresql (persistent)`. Sign-in shows OIDC.
2. **Sign-in:** an address that was not invited sees "No access". An invited Member sees only Command Centre, Scan
   Console, Scan History and System & Integrations.
3. **Scan Console:** *Benign question* gives ALLOW and a protected prompt. *Direct injection* is expected to give BLOCK,
   but no accuracy claim is made until you see it. Check that a BLOCK shows no protected prompt.
4. **WARN workflow:** on a general-profile app, input with a zero-width character (for example `note\u200b`) is held.
   An Analyst releases it, the Member resubmits it once and it is forwarded, then it is held again.
5. **Audit Log → Verify chain** shows CHAIN INTACT.
6. **Restart the app** from the Cloud menu. The data is still there, because it is in Postgres.
7. **Memory:** in Manage app → logs, check that there are no resource-limit warnings after a few scans. With a
   random-weight model of the identical architecture, the full app peaked at **1.38 GB RSS** in the build sandbox.
   This is an estimate for the real model, not a measurement of it.

## 4. Optional: run the real-model tests and the evaluation (your machine)

```powershell
$env:PPA_MODEL_DIR = "<folder containing onnx/>"
pytest tests/test_real_model.py
python eval/compare.py --model-dir $env:PPA_MODEL_DIR
```

## Zero-cost feasibility

| Item | Cost | Risk |
|---|---|---|
| Streamlit Community Cloud | $0 | Memory: an estimated 1.0–1.4 GB fits under the documented 2.7 GB cap but is above the 0.69 GB floor, so the app may be restarted under pressure. The first start is slow (model download). The app sleeps after 12 h idle. |
| Neon Free | $0 | 1 GB holds on the order of a million scan records (metadata only, roughly 1 KB each with indexes and audit events; an estimate, not measured). Waking from scale-to-zero adds latency to the first query. |
| Google / Microsoft sign-in | $0 | None for sign-in. |
| Hugging Face model download | $0 | Depends on Hugging Face being reachable from the host. |
| LLM | Not used | — |

If Community Cloud memory proves too tight, the zero-cost fallback is to run locally. No paid service is required at
any point.

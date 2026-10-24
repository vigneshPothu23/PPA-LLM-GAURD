# PPA Guard v5: setup, training, tests and deployment (Windows PowerShell)

All commands run in **PowerShell**. Paths assume the project is at `C:\PROMPT-PPA-GUARD\ppa-guard`.

## 1. Install and run locally (demo users, no accounts)

```powershell
cd C:\PROMPT-PPA-GUARD\ppa-guard
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PPAGUARD_DEMO_MODE = "1"
streamlit run streamlit_app.py
```

- The browser opens at `http://localhost:8501`. Pick a **Demo persona** in the sidebar to try each role.
- Engine 2 defaults to `protectai/deberta-v3-base-prompt-injection-v2` at a pinned revision. The first start downloads
  about 740 MB from Hugging Face (free, no account). It is labelled **EXPERIMENTAL** because it is not validated on our
  frozen test set; see `docs/v5/EVALUATION.md`.
- To use our trained alternatives instead (no download), run `$env:PPAGUARD_MODEL = "roberta-6l-experimental"` or
  `"scratch-baseline"` before `streamlit run`.

## 2. Run the tests

```powershell
pip install -r requirements-dev.txt
python -m pytest tests\v5 -q                    # v5 application tests
python -m pytest -q                             # everything, including the preserved v4/legacy suites
```

`tests\v5\test_v5_training.py` is skipped unless the training extras are installed (step 3).

## 3. Train Engine 2 (optional; CPU is enough)

You need the audited training file and the frozen evaluation sets on your PC:
- `experiments\results\injecguard_audit_v2\candidate_train.jsonl` and `kept_row_flags.jsonl`
- `eval\sets\validation.jsonl` and `eval\sets\test.jsonl`

Both locations are git-ignored, so the files are not pushed to GitHub.

```powershell
pip install -r requirements-train.txt
$env:PYTHONPATH = "src"
# a) pretrained model, fine-tuned (needs internet access to huggingface.co; about 1-2 h on CPU for 30k rows)
python -m ppaguard.engine2.train --config configs\engine2\finetune_deberta_v3_xsmall.json
# b) from scratch, no download (the configuration used for the shipped model; about 1 h on 2 CPU cores)
python -m ppaguard.engine2.train --config configs\engine2\scratch_bert_small.json
```

- Each run writes to `models\runs\<run_name>\`, which holds the log, the evaluations, the checkpoint and
  `artifact\`. A run never overwrites an existing folder.
- To try a new model in the app, run `$env:PPAGUARD_MODEL_DIR = "models\runs\<run_name>\artifact"`.
- To ship it, run `python scripts\package_artifact.py models\runs\<run_name>\artifact <name>`, then set
  `PPAGUARD_MODEL=<name>`.

To evaluate any model folder (for example ProtectAI's) on a labelled set:

```powershell
python -m ppaguard.engine2.evaluate --model-dir artifacts\engine2\roberta-6l-experimental --data eval\sets\validation.jsonl
python scripts\compare_detectors.py --model mine=models\runs\<run_name>\artifact --out models\analysis\mine
```

## 4. Put the project on GitHub

1. On github.com, create a new **empty** repository (no README, no .gitignore).
2. Push from your folder:

```powershell
cd C:\PROMPT-PPA-GUARD\ppa-guard
git init
git add .
git status                   # .venv, models\runs, eval\sets and secrets.toml must NOT be listed
git commit -m "PPA Guard v5"
git branch -M main
git remote add origin https://github.com/<your-username>/ppa-guard.git
git push -u origin main
```

If you already pushed an earlier version, unzip v5 over the same folder, then run:

```powershell
git add -A
git commit -m "PPA Guard v5"
git push
```

## 5. Deploy on Streamlit Community Cloud (free)

1. Go to share.streamlit.io and sign in with GitHub. Click **Create app**, then **Deploy a public app from GitHub**.
2. Set Repository to `<your-username>/ppa-guard`, Branch to `main`, and Main file path to `streamlit_app.py`.
3. Under **Advanced settings**, set the Python version to **3.12** and paste your **Secrets**:
   - **Demo**, quickest. Anyone with the link can pick any role, and data resets whenever the app restarts:
     ```toml
     [ppaguard]
     demo_mode = true
     ```
   - **Production:** fill in `.streamlit/secrets.toml.example`. It needs a free Neon PostgreSQL URL (use a new
     database), a Google and/or Microsoft sign-in client with redirect URI
     `https://<your-app>.streamlit.app/oauth2callback`, and your e-mail as `bootstrap_admin_email`.
4. Click **Deploy**. The first build takes a few minutes.

## 6. Check the deployment

1. **System & Integrations:**
   - Engine 2 shows `protectai/deberta-v3-base-prompt-injection-v2@89b085cd3304`, and the sidebar carries the
     EXPERIMENTAL notice.
   - Sign-in shows OIDC, or DEMO.
   - Database shows `postgresql (persistent)` in production.
2. **Engine 2 Model** page: provenance, thresholds and the dev/test metrics from the model card.
3. **Scan Console:**
   - "Benign question" should give ALLOW and a protected prompt.
   - "Direct injection" should give WARN or BLOCK.
   - "Boundary forgery" triggers the Engine 1 forgery rule.
4. **WARN flow:** as a Member, scan text that WARNs. As an Analyst, open Review Queue and click "Release once". As the
   Member again, resubmit the same text: it is forwarded once, then held again.
5. **Audit Log:** "Verify chain" shows CHAIN INTACT.
6. **Production only:** an address that was not invited sees "No access", and data survives a reboot (Manage app,
   then Reboot).

## Free-tier facts (checked 2026-10-22)

- **Streamlit Community Cloud:** up to 2.7 GB RAM per app. The docs date these limits to February 2024 and say they
  may change. Apps sleep after 12 hours without traffic, and the local filesystem is not durable.
- **Neon Free:** 1 GB storage per project, 100 CU-hours per project per month, scales to zero after 5 minutes idle.
- **Google / Microsoft sign-in:** free.
- **Engine 2 default model:** a free 740 MB download from Hugging Face on first start, with no API cost. In the build
  sandbox the whole app peaked at **1,355 MB RSS** during the browser walkthrough with this model, below the 2.7 GB cap.
  It used about 300 ms of CPU per text on 2 cores.

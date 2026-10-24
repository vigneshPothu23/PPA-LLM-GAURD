# Status update (2026-10-22): application layer

The detection core below is unchanged. On top of it, the multi-role console is now built: five roles, organization and
application scope, ALLOW/WARN/BLOCK with review, an audit chain, and OIDC sign-in. Full details are in
`docs/BUILD_REPORT.md`.
- Clean-venv test run: **283 passed, 3 skipped** (the 3 need the real checkpoint), on SQLite and PostgreSQL 16.
- Browser walkthrough: all 18 checks passed.
- Full app with a random-weight model of the official architecture: peak RSS **1.38 GB**.
- Public deployment: **not done** (no account access from the build sandbox).

---

# Status: verified vs remaining (2026-10-18)

## Verified by execution in the build sandbox
| Check | Result |
|---|---|
| Engine 1 byte-equivalence with the official SDK (commit b3c20a3), 72 separators × 3 inputs | pass |
| Policy bands (strict `>` like LLM Guard), rule floors, fail-closed on detector error | pass |
| Routing: BLOCK and default-WARN never reach Engine 1 or the LLM placeholder; ALLOW goes Engine 1 → LLM placeholder | pass |
| ONNX detector code path (tokenise, 512-token truncation flag, softmax, label index from config, checksum refusal), on a random **stub** | pass |
| Streamlit app headless: renders, routes, and fails closed when the model is missing | pass |
| Full suite in a clean venv with only `requirements.txt` + pytest | 108 passed, 3 skipped (the 3 need the real checkpoint) |
| Footprint with a **random-weight model of the exact official architecture** (184,423,682 parameters; 738,135,222-byte ONNX), 2 CPU threads, Intel Xeon 2.1 GHz | detector RSS 1.18 GB after load; full Streamlit app 1.28 GB; latency at 16/64/128/256/512 tokens = 44/99/141/297/817 ms |

## Not done, and why
| Item | Blocker | What unblocks it |
|---|---|---|
| Real DeBERTa inference and accuracy | Sandbox network policy returns 403 for huggingface.co and its file CDNs | Allow `huggingface.co`, `*.huggingface.co`, `*.hf.co`, `*.xethub.hf.co` in claude.ai Settings → Capabilities. Then `python eval/compare.py --model-dir <download>` and `PPA_MODEL_DIR=<download> pytest tests/test_real_model.py` |
| SHA-256 check of the real weights | same | The official `model.safetensors` SHA-256 is `6521cb8d0ac08148c81464899c424e6148fcc62befa371089fa4061d8b6e0424`. The ONNX file's hash must be read from its page at the pinned revision |
| Public deployment | Needs your GitHub repo and Streamlit login; this sandbox has no valid GitHub token | 5-minute manual step below, or give push access to an empty repo |

## Deploy (Streamlit Community Cloud, free)
1. Create an empty GitHub repo and push this folder. `.gitignore` already excludes models and eval data.
2. share.streamlit.io → Create app → repo, branch `main`, file `streamlit_app.py`, Python 3.12.
3. First start downloads about 750 MB from Hugging Face, so allow a few minutes.
4. Verify: *Direct injection* example → red BLOCK and "not passed to Engine 1"; *Benign question* → ALLOW with an assembled prompt.

Fit: about 1.28 GB measured against a documented 0.69–2.7 GB per-app range (Streamlit docs, Feb 2024, "may change"). Expected to fit, but not proven until deployed.
Fallbacks: (a) run locally (₹0); (b) Hugging Face Spaces: its docs now say Docker/Gradio Spaces need a paid plan to create, so that is not ₹0.

## Not claimed
- No Engine 2 accuracy, recall or FPR numbers exist yet for the real model.
- No measured security improvement from fusing the engines. What fusion adds is structural: blocked input never reaches the prompt/LLM path; rules cover cases the classifier is blind to (forged PPA separators, invisible characters, inputs over 512 tokens); a detector failure blocks.
- Engine 1's attack-reduction effect is unmeasured here because no LLM is connected.

# What to keep from earlier work (nothing has been deleted)

| Item | Where | Needed for v3? | Recommendation |
|---|---|---|---|
| Engine 1 (`ppa.py`, `official_constants.py`, equivalence tests) | protect-llm-agent-v2 | **Yes** | Copied unchanged into `src/ppa_guard/engine1` |
| Policy / pipeline design | protect-llm-agent-v2 | Yes (adapted) | Re-written for DeBERTa: strict `>` bands, LLM placeholder stage |
| TF-IDF + LR baseline, Dolly/NeurAlchemy recipe, triage, ablation | protect-llm-agent-v2 | No (evaluation comparator only) | Archive. `eval/compare.py` can still load it read-only for the comparison |
| Old Flask `app.py`, notebooks, `.pkl` files, committed datasets | github.com/vigneshPothu23/Protect-LLM-Agent | No | Archive the repo (GitHub → Settings → Archive) once v3 is live. Do not delete; it is your history |
| `C:\Protect-LLM-Agent` | your PC | No | Not accessible from here; leave untouched |
| Eval sets (NotInject, deepset, BIPIA) | rebuilt by `eval/build_eval_sets.py` from the InjecGuard and BIPIA GitHub repos | Evaluation only | Kept out of git (licences/size) |

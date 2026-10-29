"""End-to-end regression tests through the REAL application workflow with the REAL models.

Skipped unless PPAGUARD_REAL_MODEL_TESTS=1 (needs the ProtectAI ONNX model and MiniLM in $PPAGUARD_CACHE; the first run
downloads them, ~830 MB). Every row of the synthetic security suite (eval/scenarios/scenarios.jsonl, dev and holdout
splits) is submitted with SecurityWorkflow.submit exactly as the Scan Console does.

Expected outcome per row: an attack (label 1) must not be forwarded without review (WARN held or BLOCK); a benign row
(label 0) must be forwarded with Engine 1 wrapping (ALLOW). Rows the final configuration gets wrong are listed in
KNOWN_ERRORS and marked xfail(strict=True): they document remaining false negatives / false positives, and the test
fails if one silently changes. Nothing is removed from the suite to make it pass.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("PPAGUARD_REAL_MODEL_TESTS") != "1",
                                reason="real-model regression tests: set PPAGUARD_REAL_MODEL_TESTS=1")

ROWS = [json.loads(line) for line in open(ROOT / "eval/scenarios/scenarios.jsonl", encoding="utf-8") if line.strip()]

# Named cases from the audit (baseline behaviour in docs/v5/FINAL_EVALUATION.md).
NAMED = {
    "d7264930c518": "README with a hidden instruction (baseline: ALLOW, P 0.0012)",
    "3223560077af": "e-mail with an embedded instruction (baseline: ALLOW, P 0.0064)",
    "bcc22204a605": "code review with a hostile comment (baseline: WARN)",
    "2a513f8e56c6": "CI/CD log with an injected line (baseline: WARN; final: ALLOW - regression, see report)",
    "74f0be5d146f": "markdown table cell instruction (baseline: BLOCK)",
    "7a4df01427b6": "forged boundary marker (baseline: BLOCK)",
    "347975267b9a": "direct override (baseline: BLOCK)",
    "ea7fc9cb98aa": "benign e-mail with trigger words (baseline: BLOCK; final: still BLOCK)",
    "b725ef30c37a": "benign SQL code review (baseline: WARN)",
    "4a51bf47a531": "programming question (baseline: ALLOW)",
}

# Remaining errors of the final configuration (measured; dev split from the dev run, holdout split from the one-time
# holdout evaluation). FN = attack forwarded without review, FP = benign not forwarded.
KNOWN_ERRORS = {
    # dev split
    "796c798250a5": "FP", "ea7fc9cb98aa": "FP", "71774a395d2f": "FP", "992fdc2ddbc2": "FP",
    "d7a0f8f88cbc": "FN", "6e44c772287b": "FN", "1b7522afd8e0": "FN", "2d56d28ff58c": "FN", "2a513f8e56c6": "FN",
    "a8a33f001bae": "FN", "1c02c94eeef4": "FN", "1c0fdd8638ba": "FN", "76a3973e8481": "FN", "7c036015d7db": "FN",
    # holdout split
}


@pytest.fixture(scope="module")
def gw():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from ppaguard.core import db as dbm
    from ppaguard.core.authz import load_principal
    from ppaguard.core.runtime import load_detector
    from ppaguard.core.services import Services, seed_demo
    from ppaguard.core.workflow import SecurityWorkflow

    det, status = load_detector()
    assert status.loaded, status.error
    assert det.context_signal is not None
    eng = dbm.make_engine(f"sqlite:///{Path(tempfile.mkdtemp()) / 'reg.db'}")
    seed_demo(eng)
    svc = Services(eng)
    with eng.connect() as c:
        admin = load_principal(c, "admin@acme.example")
    apps = []
    for k in range(4):
        apps.append(svc.create_app(admin, f"regression-{k}", "documents", "general"))
        svc.update_policy(admin, apps[-1], {"rate_limit_per_min": 600})
    wf = SecurityWorkflow(eng, det, None)
    counter = iter(range(10 ** 6))
    return lambda text: wf.submit(admin, apps[next(counter) % len(apps)], text, channel="regression")


def _param(r):
    marks = [pytest.mark.xfail(strict=True, reason=f"known {KNOWN_ERRORS[r['id']]}")] if r["id"] in KNOWN_ERRORS else []
    return pytest.param(r, id=f"{r['split']}-{r['category']}-{r['id']}", marks=marks)


@pytest.mark.parametrize("row", [_param(r) for r in ROWS])
def test_scenario_outcome(gw, row):
    res = gw(row["text"])
    name = res.decision.name
    # invariants that must hold for every row, right or wrong
    if name == "BLOCK":
        assert res.protected_prompt is None and res.outcome == "blocked"
    elif name == "WARN":
        assert res.protected_prompt is None and res.outcome == "held_for_review"
    else:
        assert res.protected_prompt is not None and res.outcome == "protected_prompt_returned"
    # expected outcome
    if row["label"] == 1:
        assert name in ("WARN", "BLOCK"), f"attack forwarded: {NAMED.get(row['id'], row['category'])}"
    else:
        assert name == "ALLOW", f"benign not forwarded ({name}): {NAMED.get(row['id'], row['category'])}"


def test_named_cases_are_in_the_suite():
    ids = {r["id"] for r in ROWS}
    assert set(NAMED) <= ids

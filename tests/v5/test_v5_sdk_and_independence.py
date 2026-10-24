"""Python integration (SDK + the runnable example) and the independence of the v5 namespace."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("onnxruntime")
from sqlalchemy import select  # noqa: E402

from ppaguard.core import db as dbm  # noqa: E402
from ppaguard.core.authz import Forbidden, load_principal  # noqa: E402
from ppaguard.core.services import Services, seed_demo  # noqa: E402
from ppaguard.sdk import PPAGuardClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/v5/fixtures/test_model"


@pytest.fixture
def setup(tmp_path):
    url = f"sqlite:///{tmp_path / 'sdk.db'}"
    eng = dbm.make_engine(url)
    seed_demo(eng)
    s = Services(eng)
    with eng.connect() as c:
        dev = load_principal(c, "dev@acme.example")
    app = next(a for a in s.list_apps(dev) if a["name"] == "code-review-bot")
    key = s.create_api_key(dev, app["id"])
    return {"url": url, "eng": eng, "s": s, "dev": dev, "app": app, "key": key}


def test_example_script_runs_through_the_full_workflow(setup):
    env = dict(os.environ, PPAGUARD_API_KEY=setup["key"], PPAGUARD_DATABASE_URL=setup["url"], PPAGUARD_MODEL_DIR=str(FIX))
    out = subprocess.run([sys.executable, str(ROOT / "examples/integration_example.py")], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("decision:") == 3 and "TEST-ONLY" in out.stdout
    assert "boundary forgery" in out.stdout                       # the <|im_start|> example is caught by the rule
    with setup["eng"].connect() as c:
        rows = c.execute(select(dbm.scans.c.channel, dbm.scans.c.app_id, dbm.scans.c.api_key_id)).all()
    assert len(rows) == 3 and all(r[0] == "sdk" and r[1] == setup["app"]["id"] and r[2] for r in rows)


def test_client_requires_a_valid_unrevoked_key(setup, monkeypatch):
    monkeypatch.setenv("PPAGUARD_MODEL_DIR", str(FIX))
    with pytest.raises(ValueError):
        PPAGuardClient(api_key="", database_url=setup["url"])
    with pytest.raises(Forbidden):
        PPAGuardClient(api_key="ppag_forged", database_url=setup["url"]).protect("hello")
    good = PPAGuardClient(api_key=setup["key"], database_url=setup["url"])
    r = good.protect("Please review this function.")
    assert r.scan_id and r.decision in ("ALLOW", "WARN", "BLOCK")
    if r.allowed:
        assert not good.check_response("fine", r) and good.check_response(f"echo {r._separator.open}", r)
    kid = setup["s"].list_api_keys(setup["dev"], setup["app"]["id"])[0]["id"]
    setup["s"].revoke_api_key(setup["dev"], kid)
    with pytest.raises(Forbidden):
        good.protect("hello again")


def test_client_without_model_blocks_everything(setup, monkeypatch, tmp_path):
    monkeypatch.setenv("PPAGUARD_MODEL_DIR", str(tmp_path / "nowhere"))
    c = PPAGuardClient(api_key=setup["key"], database_url=setup["url"])
    r = c.protect("Please review this function.")
    assert r.decision == "BLOCK" and not r.allowed and r.protected_prompt is None and c.model_id is None


def test_paused_application_refuses_sdk_calls(setup, monkeypatch):
    monkeypatch.setenv("PPAGUARD_MODEL_DIR", str(FIX))
    setup["s"].set_app_status(setup["dev"], setup["app"]["id"], "paused")
    with pytest.raises(Forbidden):
        PPAGuardClient(api_key=setup["key"], database_url=setup["url"]).protect("hello")


def test_v5_namespace_never_imports_the_old_implementation():
    offenders = []
    for path in (ROOT / "src/ppaguard").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(n == "ppa_guard" or n.startswith("ppa_guard.") for n in names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    for path in [ROOT / "streamlit_app.py", *(ROOT / "src/ppaguard/ui").rglob("*.py"), ROOT / "examples/integration_example.py"]:
        if path.exists() and "ppa_guard" in path.read_text(encoding="utf-8").replace("ppa_guard.db", ""):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_v5_runs_with_old_package_unimportable(tmp_path):
    """Hide the old package entirely and exercise the v5 workflow in a fresh interpreter."""
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'ppa_guard' or name.startswith('ppa_guard.'): raise ImportError('old package blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "from ppaguard.core import db; from ppaguard.core.services import Services, seed_demo\n"
        "from ppaguard.core.authz import load_principal; from ppaguard.core.workflow import SecurityWorkflow\n"
        "from ppaguard.engine2 import Detector\n"
        f"e = db.make_engine('sqlite:///{tmp_path / 'i.db'}'); seed_demo(e)\n"
        "c = e.connect(); p = load_principal(c, 'member@acme.example'); c.close()\n"
        "app = [a for a in Services(e).list_apps(p) if a['name'] == 'mail-assistant'][0]\n"
        f"r = SecurityWorkflow(e, Detector(r'{FIX}')).submit(p, app['id'], 'Summarise this note.')\n"
        "print('OK', r.decision.name, r.engine2['model_id'])\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    assert out.returncode == 0 and out.stdout.startswith("OK"), out.stderr

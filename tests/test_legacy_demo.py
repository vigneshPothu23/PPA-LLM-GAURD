"""Legacy single-page demo (kept runnable): headless Streamlit test with the random STUB model."""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("onnxruntime")
ROOT = Path(__file__).resolve().parents[1]


def _app(monkeypatch, model_dir):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PPA_MODEL_DIR", str(model_dir))
    return AppTest.from_file(str(ROOT / "legacy/single_page_demo.py"), default_timeout=60)


def test_app_renders_and_routes_with_stub(monkeypatch):
    at = _app(monkeypatch, ROOT / "tests/fixtures/stub_model").run()
    assert not at.exception and not at.error
    at.text_area(key="prompt").set_value("Write a haiku about autumn.")
    at.button[-1].click().run()
    assert not at.exception
    decisions = [e.value for e in [*at.success, *at.warning, *at.error] if "Decision:" in e.value]
    assert len(decisions) == 1


def test_app_fails_closed_when_model_missing(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()  # empty folder -> load error
    assert any("could not be loaded" in e.value for e in at.error)
    at.text_area(key="prompt").set_value("hello")
    at.button[-1].click().run()
    assert any("Decision: BLOCK" in e.value for e in at.error)
    assert not at.code  # nothing assembled

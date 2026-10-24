"""[LEGACY v4 console] Headless tests of the Streamlit console (demo mode + random STUB model: UI wiring and authorization, not detection
quality). The server-side rules themselves are tested in test_platform.py."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("onnxruntime")
pytest.importorskip("sqlalchemy")
ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "tests/fixtures/stub_model"

PERSONAS = {
    "admin@acme.example": {"command-centre", "scan", "reviews", "history", "applications", "members", "audit", "system"},
    "analyst@acme.example": {"command-centre", "scan", "reviews", "history", "applications", "audit", "system"},
    "dev@acme.example": {"command-centre", "scan", "history", "applications", "system"},
    "auditor@acme.example": {"command-centre", "history", "applications", "members", "audit", "system"},
    "member@acme.example": {"command-centre", "scan", "history", "system"},
}


@pytest.fixture
def app(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.setenv("PPA_DEMO_MODE", "1")
    monkeypatch.setenv("PPA_DEMO_DB", str(tmp_path / "demo.db"))
    monkeypatch.setenv("PPA_MODEL_DIR", str(STUB))
    monkeypatch.delenv("PPA_DATABASE_URL", raising=False)

    def make(persona=None, model_dir=STUB):
        monkeypatch.setenv("PPA_MODEL_DIR", str(model_dir))
        at = AppTest.from_file(str(ROOT / "legacy/console_v4_app.py"), default_timeout=60).run()
        if persona:
            at.selectbox(key="persona").set_value(persona).run()
        return at
    yield make
    st.cache_resource.clear()


def _pages(at):
    return {info.get("url_pathname") or "command-centre" for info in at._registered_pages.values()}  # default page: ""


def _goto(at, url_path):
    h = next(k for k, info in at._registered_pages.items() if (info.get("url_pathname") or "command-centre") == url_path)
    at._page_hash = h
    return at.run()


def _text(at):
    return " ".join(str(m.value) for m in [*at.markdown, *at.caption, *at.info, *at.success, *at.warning, *at.error])


def test_unconfigured_sign_in_fails_closed(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    monkeypatch.delenv("PPA_DEMO_MODE", raising=False)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(ROOT / "legacy/console_v4_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("not configured" in e.value for e in at.error)
    assert not at.dataframe and not at.text_area


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_each_role_sees_only_its_pages_and_every_page_renders(app, persona):
    at = app(persona)
    assert _pages(at) == PERSONAS[persona]
    for page in sorted(PERSONAS[persona]):
        at = _goto(at, page)
        assert not at.exception, (persona, page, at.exception)
        assert not [e.value for e in at.error if "Not permitted" in e.value], (persona, page)
    assert "DEMO MODE" in _text(at)


def test_member_scan_produces_recorded_decision_and_ppa_output(app):
    at = _goto(app("member@acme.example"), "scan")
    at.text_area(key="scan_text").set_value("Write a haiku about autumn.").run()
    at.button(key="scan_go").click().run()
    assert not at.exception
    t = _text(at)
    assert any(d in t for d in ("ALLOW", "WARN", "BLOCK")) and "record " in t
    at = _goto(at, "history")
    assert at.dataframe and not at.exception


def test_missing_model_blocks_every_scan(app, tmp_path):
    at = app("member@acme.example", model_dir=tmp_path / "empty")
    assert any("fail closed" in e.value for e in at.sidebar.error)
    at = _goto(at, "scan")
    at.text_area(key="scan_text").set_value("hello").run()
    at.button(key="scan_go").click().run()
    t = _text(at)
    assert "BLOCK" in t and "fail closed" in t
    assert not at.code  # no protected prompt assembled


def test_member_cannot_reach_admin_page_by_url(app):
    at = app("member@acme.example")
    assert "members" not in _pages(at) and "audit" not in _pages(at) and "reviews" not in _pages(at)
    with pytest.raises(StopIteration):
        _goto(at, "members")


def _pick_app(at, name):
    sb = at.selectbox(key="scan_app")
    sb.select_index(next(i for i, o in enumerate(sb.options) if o == name)).run()
    return at


def test_warn_held_then_analyst_release_is_used_once(app):
    """End-to-end WARN workflow through the UI. The STUB model scores 0.50 (not > 0.50, so ALLOW band); the WARN comes
    from the general profile's hidden-character rule, so the outcome is deterministic."""
    txt = "Please summarise this note.\u200b"
    at = _pick_app(_goto(app("member@acme.example"), "scan"), "mail-assistant")
    at.text_area(key="scan_text").set_value(txt).run()
    at.button(key="scan_go").click().run()
    assert "Held for a Security Analyst" in _text(at) and not at.code

    at.selectbox(key="persona").set_value("analyst@acme.example").run()
    at = _goto(at, "reviews")
    release = [b for b in at.button if b.label == "Release once"]
    assert len(release) == 1
    release[0].click().run()
    assert not at.exception and not [b for b in at.button if b.label == "Release once"]

    at.selectbox(key="persona").set_value("member@acme.example").run()
    at = _pick_app(_goto(at, "scan"), "mail-assistant")
    at.text_area(key="scan_text").set_value(txt).run()
    at.button(key="scan_go").click().run()
    assert "analyst release used" in _text(at) and at.code      # forwarded once, with the PPA-protected prompt
    at.button(key="scan_go").click().run()
    assert "Held for a Security Analyst" in _text(at)            # single use

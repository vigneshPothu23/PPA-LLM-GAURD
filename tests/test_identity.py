"""Rule 1 configuration logic: which sign-in mode is active, which identity claims are accepted, first-admin bootstrap."""

import pytest

pytest.importorskip("sqlalchemy")
from ppa_guard.platform import db as dbm  # noqa: E402
from ppa_guard.platform import identity  # noqa: E402
from ppa_guard.platform.authz import Forbidden, load_principal  # noqa: E402
from ppa_guard.platform.roles import Role  # noqa: E402
from ppa_guard.platform.runtime import load_detector  # noqa: E402
from ppa_guard.platform.services import Services  # noqa: E402

GOOGLE = {"auth": {"redirect_uri": "https://x/oauth2callback", "cookie_secret": "s", "client_id": "id", "client_secret": "c",
                   "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration"}}
NAMED = {"auth": {"redirect_uri": "r", "cookie_secret": "s",
                  "google": {"client_id": "a", "client_secret": "b", "server_metadata_url": "u"},
                  "microsoft": {"client_id": "c", "client_secret": "d", "server_metadata_url": "v"}}}


def test_mode_selection_is_fail_closed():
    assert identity.auth_mode({}, {}) == "unconfigured"
    assert identity.auth_mode({"auth": {"redirect_uri": "r"}}, {}) == "unconfigured"   # incomplete config
    assert identity.auth_mode(GOOGLE, {}) == "oidc"
    assert identity.auth_mode(NAMED, {}) == "oidc"
    assert identity.oidc_providers(NAMED) == ["google", "microsoft"] and identity.oidc_providers(GOOGLE) == []
    assert identity.auth_mode({}, {"PPA_DEMO_MODE": "1"}) == "demo"
    assert identity.auth_mode({"ppa": {"demo_mode": True}}, {}) == "demo"
    assert identity.auth_mode({}, {"PPA_DEMO_MODE": "0"}) == "unconfigured"


def test_secrets_object_that_raises_is_treated_as_empty():
    class Raising:
        def get(self, *a, **k):
            raise FileNotFoundError("no secrets.toml")
    assert identity.auth_mode(Raising(), {}) == "unconfigured"


@pytest.mark.parametrize("claims,expected", [
    ({"is_logged_in": True, "email": "Ana@Example.com", "email_verified": True}, "ana@example.com"),
    ({"is_logged_in": True, "email": "ana@example.com"}, "ana@example.com"),                    # Microsoft: no claim
    ({"is_logged_in": True, "preferred_username": "ana@contoso.com"}, "ana@contoso.com"),
    ({"is_logged_in": True, "email": "ana@example.com", "email_verified": False}, None),        # unverified address
    ({"is_logged_in": False, "email": "ana@example.com"}, None),
    ({"is_logged_in": True, "email": "not-an-email"}, None),
    ({}, None),
])
def test_verified_email(claims, expected):
    assert identity.verified_email(claims) == expected


MS = "https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/v2.0"


def test_microsoft_identity_needs_an_allowed_tenant():
    claims = {"is_logged_in": True, "email": "ana@contoso.com", "iss": MS, "tid": "11111111-2222-3333-4444-555555555555"}
    assert identity.verified_email(claims) is None                                     # no tenant allowed
    assert identity.verified_email(claims, ("99999999-0000-0000-0000-000000000000",)) is None
    assert identity.verified_email(claims, identity.allowed_microsoft_tenants(
        {"ppa": {"microsoft_tenant_ids": ["11111111-2222-3333-4444-555555555555"]}})) == "ana@contoso.com"
    assert identity.allowed_microsoft_tenants({"ppa": {"microsoft_tenant_ids": "ABC"}}) == ("abc",)


def test_bootstrap_creates_first_admin_once_and_nobody_else(tmp_path):
    eng = dbm.make_engine(f"sqlite:///{tmp_path / 'b.db'}")
    s = Services(eng)
    sec = {"ppa": {"bootstrap_admin_email": "Owner@Corp.com", "bootstrap_org_name": "Corp", "bootstrap_industry": "finance"}}
    assert identity.bootstrap(eng, s, {}) is None                 # nothing configured -> nothing created
    org = identity.bootstrap(eng, s, sec)
    assert org and identity.bootstrap(eng, s, sec) is None        # idempotent
    with eng.connect() as c:
        p = load_principal(c, "owner@corp.com")
        assert p.role is Role.ADMIN and p.org_id == org
        with pytest.raises(Forbidden):
            load_principal(c, "someone.else@corp.com")            # not invited -> no access


def test_detector_loader_reports_failure_and_never_substitutes(tmp_path):
    det, status = load_detector(str(tmp_path / "missing"))
    assert det is None and not status.loaded and status.error and status.source == "none"


def test_detector_loader_labels_local_directory_unverified():
    pytest.importorskip("onnxruntime")
    from pathlib import Path
    det, status = load_detector(str(Path(__file__).parent / "fixtures/stub_model"))
    assert det is not None and status.source == "local-directory" and "not verified" in status.verified


def test_detector_loader_rejects_checksum_mismatch(monkeypatch):
    pytest.importorskip("onnxruntime")
    from pathlib import Path
    monkeypatch.setenv("PPA_MODEL_SHA256", "0" * 64)
    det, status = load_detector(str(Path(__file__).parent / "fixtures/stub_model"))
    assert det is None and "checksum mismatch" in status.error

"""Tests for the WeatherNext local exporter that need no credentials.

The query itself can only be checked against a real table, but the two parts
that decide whether the output is usable — column discovery and the JSON/grid
contract that blend_live.py enforces — are checkable here, and this is the last
place they can be checked before the script leaves for someone else's machine.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "wn", Path(__file__).resolve().parents[1] / "scripts" / "wn_export_local.py")
wn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wn)


def test_discovery_maps_the_obvious_schema():
    cols = wn.discover(["init_time", "lead_time", "ensemble_member",
                        "10m_u_component_of_wind", "10m_v_component_of_wind",
                        "latitude", "longitude", "temperature_2m"])
    assert cols["init"] == "init_time"
    assert cols["u10"] == "10m_u_component_of_wind"
    assert cols["v10"] == "10m_v_component_of_wind"
    assert cols["lat"] == "latitude"


def test_discovery_accepts_the_dynamical_style_names():
    cols = wn.discover(["forecast_reference_time", "step", "realization",
                        "wind_u_10m", "wind_v_10m", "lat", "lon"])
    assert cols["u10"] == "wind_u_10m" and cols["member"] == "realization"


def test_missing_wind_column_is_named_not_guessed():
    with pytest.raises(SystemExit, match="u10"):
        wn.discover(["init_time", "lead_time", "latitude", "longitude",
                     "temperature_2m"])


def test_member_is_optional_for_a_deterministic_table():
    cols = wn.discover(["init_time", "lead_time", "u10", "v10", "lat", "lon"])
    assert "member" not in cols          # absent, and that is allowed
    assert cols["u10"] == "u10"


def test_ambiguous_wind_columns_refuse_rather_than_pick():
    """Two plausible u columns must stop the run, not silently take one."""
    with pytest.raises(SystemExit, match="ambiguous"):
        wn.discover(["init_time", "lead_time", "latitude", "longitude",
                     "u_component_of_wind_80m", "u_component_of_wind_120m",
                     "v_comp_10m"])


def test_json_matches_the_grid_blend_live_requires():
    """The contract: blend_live.py refuses anything whose header differs."""
    n = wn.NX * wn.NY
    rec = wn.velocity_records([1.5] * n, [-2.5] * n, "2026-08-10T12:00:00Z", 72)
    u, v = rec
    assert u["header"]["nx"] == 180 and u["header"]["ny"] == 90
    assert u["header"]["la1"] == 89.125 and u["header"]["lo1"] == 0.875
    assert u["header"]["dx"] == 2.0 and u["header"]["dy"] == 2.0
    assert u["header"]["parameterNumber"] == 2      # u first
    assert v["header"]["parameterNumber"] == 3      # then v
    assert u["header"]["forecastTime"] == 72
    assert len(u["data"]) == n and len(v["data"]) == n
    assert u["data"][0] == 1.5 and v["data"][0] == -2.5


def test_grid_header_is_identical_to_the_existing_live_sources():
    """Byte-compatible with fetch_dynamical.py's export, or the blend refuses."""
    import json
    live = Path("viewer/data/aifs_live_latest_072.json")
    if not live.exists():
        pytest.skip("no live sample committed")
    ref = json.loads(live.read_text())[0]["header"]
    got = wn.velocity_records([0.0] * (wn.NX * wn.NY), [0.0] * (wn.NX * wn.NY),
                              ref["refTime"], 72)[0]["header"]
    for k in ("nx", "ny", "la1", "lo1", "la2", "lo2", "dx", "dy", "forecastTime"):
        assert got[k] == ref[k], f"{k}: {got[k]} != {ref[k]} (blend would refuse)"


# --- who we authenticate AS -------------------------------------------------
#
# WeatherNext access was granted to a service account, and the exporter
# authenticates as whoever ran `gcloud auth application-default login`. Those
# can be different principals, and when they are, the failure is a 403 that
# looks exactly like "the dataset is not shared with you". These tests pin the
# wiring that makes the difference visible, and they run without credentials or
# network -- which is the only place this part can be checked before the script
# leaves for someone else's machine.

SA = "631486859154-compute@developer.gserviceaccount.com"


class _FakeSource:
    """Enough of a credential for impersonated_credentials to wrap, no more."""
    universe_domain = "googleapis.com"


def test_no_impersonation_uses_adc_unchanged(monkeypatch):
    google_auth = pytest.importorskip("google.auth")
    source = _FakeSource()
    monkeypatch.setattr(google_auth, "default", lambda scopes=None: (source, "proj"))
    creds, impersonating = wn.build_credentials(None)
    assert creds is source
    assert impersonating is None


def test_impersonation_builds_the_credential_offline(monkeypatch):
    """Constructing it must not touch the network -- the token is minted lazily."""
    google_auth = pytest.importorskip("google.auth")
    from google.auth import impersonated_credentials
    monkeypatch.setattr(google_auth, "default",
                        lambda scopes=None: (_FakeSource(), "proj"))
    creds, impersonating = wn.build_credentials(SA)
    assert isinstance(creds, impersonated_credentials.Credentials)
    assert impersonating == SA
    assert wn.principal_of(creds) == SA          # printed before anything else


def test_principal_is_reported_not_guessed():
    class Anonymous:                     # a user ADC token carries no email
        pass
    assert "unknown" in wn.principal_of(Anonymous())


def test_denied_without_impersonation_names_the_token_creator_role():
    msg = wn.denied_message("me@example.com", None, RuntimeError("403"))
    assert "me@example.com" in msg
    assert "roles/iam.serviceAccountTokenCreator" in msg
    assert "--impersonate" in msg


def test_denied_while_impersonating_blames_the_data_grant_instead():
    """The two 403s have different fixes; conflating them sends you in circles."""
    msg = wn.denied_message(SA, SA, RuntimeError("403"))
    assert "roles/bigquery.dataViewer" in msg
    assert "roles/iam.serviceAccountTokenCreator" not in msg


def test_guarded_turns_a_403_into_the_explanation():
    class Forbidden(Exception):
        pass

    def boom():
        raise Forbidden("Access Denied: Table x")

    with pytest.raises(SystemExit, match="serviceAccountTokenCreator"):
        wn.guarded(boom, "me@example.com", None)


def test_guarded_reraises_anything_that_is_not_an_iam_problem():
    def boom():
        raise ValueError("bad SQL")

    with pytest.raises(ValueError):
        wn.guarded(boom, "me@example.com", None)


def test_missing_credentials_is_an_instruction_not_a_traceback(monkeypatch):
    google_auth = pytest.importorskip("google.auth")

    class DefaultCredentialsError(Exception):
        pass

    def no_creds(scopes=None):
        raise DefaultCredentialsError("not found")

    monkeypatch.setattr(google_auth, "default", no_creds)
    with pytest.raises(SystemExit, match="application-default login"):
        wn.build_credentials(None)

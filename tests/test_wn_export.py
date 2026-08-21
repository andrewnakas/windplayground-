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
                        "latitude", "longitude", "temperature_2m"])["cols"]
    assert cols["init"] == "init_time"
    assert cols["u10"] == "10m_u_component_of_wind"
    assert cols["v10"] == "10m_v_component_of_wind"
    assert cols["lat"] == "latitude"


def test_discovery_accepts_the_dynamical_style_names():
    cols = wn.discover(["forecast_reference_time", "step", "realization",
                        "wind_u_10m", "wind_v_10m", "lat", "lon"])["cols"]
    assert cols["u10"] == "wind_u_10m" and cols["member"] == "realization"


def test_missing_wind_column_is_named_not_guessed():
    with pytest.raises(SystemExit, match="u10"):
        wn.discover(["init_time", "lead_time", "latitude", "longitude",
                     "temperature_2m"])


def test_member_is_optional_for_a_deterministic_table():
    cols = wn.discover(["init_time", "lead_time", "u10", "v10", "lat", "lon"])["cols"]
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
    live = Path("docs/data/aifs_live_latest_072.json")
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


# --- the schema WeatherNext actually has ------------------------------------
#
# Verified against Google's BigQuery and Earth Engine docs, not guessed: the
# wind lives inside a REPEATED RECORD called `forecast`, and position is a
# GEOGRAPHY point rather than a latitude/longitude pair. The first version of
# this exporter assumed flat columns and would have exited "could not find
# columns for u10, v10, lat, lon" against a table containing all four -- a
# failure indistinguishable from a missing grant.

class _F:
    """Minimal stand-in for google.cloud.bigquery.SchemaField."""
    def __init__(self, name, field_type, mode="NULLABLE", fields=()):
        self.name, self.field_type, self.mode, self.fields = (
            name, field_type, mode, fields)


WEATHERNEXT_SCHEMA = [
    _F("init_time", "TIMESTAMP"),
    _F("geography", "GEOGRAPHY"),
    _F("geography_polygon", "GEOGRAPHY"),
    _F("forecast", "RECORD", "REPEATED", fields=[
        _F("hours", "INTEGER"),
        _F("time", "TIMESTAMP"),
        _F("2m_temperature", "FLOAT"),
        _F("10m_u_component_of_wind", "FLOAT"),
        _F("10m_v_component_of_wind", "FLOAT"),
        _F("total_precipitation", "FLOAT"),
    ]),
]


def test_flatten_reaches_inside_the_repeated_record():
    """A probe that stops at 'forecast RECORD' cannot do the job it exists for."""
    flat = wn.flatten_schema(WEATHERNEXT_SCHEMA)
    paths = [p for p, _, _ in flat]
    assert "forecast.10m_u_component_of_wind" in paths
    assert ("forecast", "RECORD", "REPEATED") in flat


def test_discovery_finds_wind_nested_and_position_in_the_geography():
    lay = wn.discover(wn.flatten_schema(WEATHERNEXT_SCHEMA))
    assert lay["kind"] == "nested"
    assert lay["unnest"] == "forecast"
    assert lay["geo"] == "geography"           # NOT geography_polygon
    assert lay["cols"]["u10"] == "forecast.10m_u_component_of_wind"
    assert lay["cols"]["lead"] == "forecast.hours"
    assert lay["cols"]["init"] == "init_time"


def test_flat_tables_still_report_a_flat_layout():
    """Regression: the simple schema must not be dragged through the new path."""
    lay = wn.discover(["init_time", "lead_time", "u10", "v10", "lat", "lon"])
    assert lay["kind"] == "flat" and lay["unnest"] is None and lay["geo"] is None


def _wn_sql(leads=(0, 24, 48, 72, 96, 120)):
    lay = wn.discover(wn.flatten_schema(WEATHERNEXT_SCHEMA))
    return wn.grid_sql(lay, "p.d.weathernext_2_0_0_mean",
                       "2026-08-18T00:00:00Z", list(leads), "INTEGER")


def test_digit_leading_identifiers_are_backticked():
    """`10m_u_component_of_wind` unquoted is a BigQuery syntax error, not a typo."""
    sql = _wn_sql()
    assert "t2.`10m_u_component_of_wind`" in sql
    assert "t2.`10m_v_component_of_wind`" in sql


def test_position_comes_from_the_point_not_the_polygon():
    sql = _wn_sql()
    assert "ST_Y(t1.`geography`)" in sql and "ST_X(t1.`geography`)" in sql
    assert "geography_polygon" not in sql


def test_every_lead_is_fetched_in_one_query():
    """Per-lead queries re-scan the nested array and bill six times over."""
    sql = _wn_sql()
    assert sql.count("UNNEST") == 1
    assert "IN (0, 24, 48, 72, 96, 120)" in sql
    assert "GROUP BY lead_h, gy, gx" in sql


def test_flat_layout_generates_no_unnest():
    lay = wn.discover(["init_time", "lead_time", "u10", "v10", "lat", "lon"])
    sql = wn.grid_sql(lay, "p.d.t", "2026-08-18T00:00:00Z", [24], "INTEGER")
    assert "UNNEST" not in sql and "ST_X" not in sql


def test_the_longitude_wrap_in_the_sql_is_the_one_we_mean():
    """ST_X returns -180..180; the grid starts at 0.875 and must wrap, not clamp.

    The formula is duplicated here on purpose: the test above pins that this
    exact expression reaches the SQL, and this one pins that the expression is
    correct. Neither check is worth much without the other.
    """
    assert "MOD(CAST(FLOOR((ST_X(t1.`geography`) - 0.875) / 2.0) AS INT64) + 180, 180)" \
        in _wn_sql()

    import math as _m

    def gx(lon):
        return (_m.floor((lon - wn.LON0) / wn.DX) + wn.NX) % wn.NX

    assert gx(0.875) == 0                    # first cell, at its west edge
    assert gx(-90.0) == 134                  # a NEGATIVE lon lands at 270 E
    assert gx(-0.1) == 179                   # just west of Greenwich -> last cell
    # 179.9 E and -179.9 E are the SAME cell: it spans 178.875..180.875 E, so it
    # straddles the date line. Wrapping puts them together; clamping would not.
    assert gx(179.9) == gx(-179.9) == 89
    assert len({gx(-180 + i * 2.0) for i in range(180)}) == 180   # a bijection


# --- matching the init the blend will accept --------------------------------
#
# blend_live.py refuses members whose refTime differs, so "WeatherNext's newest
# init" is the wrong default -- the export succeeds and the blend then refuses,
# with nothing in either output pointing at why.

def _manifest(tmp_path, sources):
    import json
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": sources}))
    return tmp_path


def test_match_live_returns_the_init_the_others_carry(tmp_path):
    d = _manifest(tmp_path, [
        {"id": "aifs_live", "kind": "live", "init_time": "2026-08-17T18:00:00Z"},
        {"id": "gfs_live", "kind": "live", "init_time": "2026-08-17T18:00:00Z"},
        {"id": "graphcast_2020", "kind": "hindcast"},
    ])
    assert wn.live_init(d) == "2026-08-17T18:00:00Z"


def test_match_live_ignores_the_blend_and_weathernext_itself(tmp_path):
    """The blend is derived from the members, and our own row may be stale."""
    d = _manifest(tmp_path, [
        {"id": "aifs_live", "kind": "live", "init_time": "2026-08-18T00:00:00Z"},
        {"id": "gfs_live", "kind": "live", "init_time": "2026-08-18T00:00:00Z"},
        {"id": "live_blend", "kind": "live", "init_time": "2026-08-10T12:00:00Z"},
        {"id": wn.SOURCE_ID, "kind": "live", "init_time": "2026-08-01T00:00:00Z"},
    ])
    assert wn.live_init(d) == "2026-08-18T00:00:00Z"


def test_match_live_refuses_when_the_existing_members_disagree(tmp_path):
    d = _manifest(tmp_path, [
        {"id": "aifs_live", "kind": "live", "init_time": "2026-08-18T00:00:00Z"},
        {"id": "gfs_live", "kind": "live", "init_time": "2026-08-17T18:00:00Z"},
    ])
    with pytest.raises(SystemExit, match="disagree"):
        wn.live_init(d)


def test_match_live_says_what_to_do_when_there_are_no_live_members(tmp_path):
    d = _manifest(tmp_path, [{"id": "graphcast_2020", "kind": "hindcast"}])
    with pytest.raises(SystemExit, match="fetch_dynamical"):
        wn.live_init(d)

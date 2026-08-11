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

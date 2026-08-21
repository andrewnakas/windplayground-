"""Lead selection for the live fetch: request order kept, absences tolerated.

Pure-function test; the fetch itself needs zarr>=3 and network, but which
leads survive against a store's inventory must not.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

# fetch_dynamical imports xarray/zarr at module top; lift just the two
# module-level names this test targets instead of importing the world.
SRC = (Path(__file__).resolve().parents[1] / "scripts" / "fetch_dynamical.py").read_text()
tree = ast.parse(SRC)
wanted = [n for n in tree.body
          if (isinstance(n, ast.FunctionDef) and n.name == "select_leads")
          or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DEFAULT_LEADS")]
mod = types.ModuleType("fetch_leads_under_test")
exec(compile(ast.Module(body=wanted, type_ignores=[]), "<fetch_dynamical>", "exec"),
     mod.__dict__)
select_leads, DEFAULT_LEADS = mod.select_leads, mod.DEFAULT_LEADS


def test_default_is_the_full_six_hourly_ladder():
    assert DEFAULT_LEADS == list(range(0, 121, 6))
    assert len(DEFAULT_LEADS) == 21


def test_absent_leads_are_dropped_not_fatal():
    # a store that starts at +6 (WeatherNext-shaped) just loses lead 0
    assert select_leads([0, 6, 12], {6, 12, 18}) == [6, 12]


def test_request_order_is_preserved():
    assert select_leads([12, 0, 6], {0, 6, 12}) == [12, 0, 6]


def test_nothing_available_is_fatal():
    with pytest.raises(SystemExit, match="none of"):
        select_leads([0, 6], {24, 48})

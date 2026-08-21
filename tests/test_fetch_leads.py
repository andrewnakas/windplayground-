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


def test_default_is_the_full_hourly_ladder():
    assert DEFAULT_LEADS == list(range(0, 121))
    assert len(DEFAULT_LEADS) == 121


def test_six_hourly_store_keeps_its_native_cadence():
    # AIFS publishes 6-hourly; the hourly request intersects down to 21 rungs
    assert select_leads(DEFAULT_LEADS, set(range(0, 121, 6))) ==         list(range(0, 121, 6))


def test_absent_leads_are_dropped_not_fatal():
    # a store that starts at +6 (WeatherNext-shaped) just loses lead 0
    assert select_leads([0, 6, 12], {6, 12, 18}) == [6, 12]


def test_request_order_is_preserved():
    assert select_leads([12, 0, 6], {0, 6, 12}) == [12, 0, 6]


def test_nothing_available_is_fatal():
    with pytest.raises(SystemExit, match="none of"):
        select_leads([0, 6], {24, 48})


# --- bin_regrid: projected regional grid -> regular lat/lon ------------------

import numpy as np

wanted2 = [n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "bin_regrid"]
mod2 = types.ModuleType("regrid_under_test")
mod2.np = np
exec(compile(ast.Module(body=wanted2, type_ignores=[]), "<fetch_dynamical>", "exec"),
     mod2.__dict__)
bin_regrid = mod2.bin_regrid


def test_bin_regrid_averages_into_cells_north_first():
    # a flat "projection": 4 points per 1-degree cell over a 2x2-cell box,
    # value = 10*row + col of the cell the point belongs to
    pts_lat, pts_lon, vals = [], [], []
    for cy in range(2):
        for cx in range(2):
            for oy in (0.25, 0.75):
                for ox in (0.25, 0.75):
                    pts_lat.append(10 + cy + oy)
                    pts_lon.append(20 + cx + ox)
                    vals.append(10 * cy + cx)
    lat, lon, (out,) = bin_regrid(np.array(pts_lat), np.array(pts_lon),
                                  [np.array(vals, dtype=float)], 1.0)
    assert out.shape == (2, 2) and len(lat) == 2 and len(lon) == 2
    assert lat[0] > lat[-1]                     # north first
    assert np.isfinite(out).all()               # full coverage after crop
    # row 0 is NORTH (cy=1): cells 10,11; row 1 is south: 0,1
    assert out.tolist() == [[10.0, 11.0], [0.0, 1.0]]


def test_bin_regrid_crops_uncovered_corners():
    # points fill a diamond; the corners of the bounding box are empty and
    # must be cropped away rather than exported as holes
    n = 41
    y, x = np.mgrid[0:n, 0:n]
    keep = (abs(x - n // 2) + abs(y - n // 2)) <= n // 2
    lat2d = (10 + y * 0.1)[keep]
    lon2d = (20 + x * 0.1)[keep]
    f = np.ones(keep.sum())
    lat, lon, (out,) = bin_regrid(lat2d, lon2d, [f], 0.2)
    assert out.size > 0
    assert np.isfinite(out).all()

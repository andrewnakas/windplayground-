"""Tests for build_site.export_metrics — the viewer's only source of scores.

The exporter is the seam between artifacts/results/ and the dashboard: if it
misaligns a lead array or lets a NaN through, the skill panel quietly lies.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))          # build_site imports windml.config

SPEC = importlib.util.spec_from_file_location(
    "build_site", REPO / "scripts" / "build_site.py")
bs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bs)

CSV_HEADER = "model,variable,lead_h,rmse,acc,n_inits,git_sha,split\n"


def write_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True)
    # a plain model: scores at 24 and 72, a NaN at 48, an unscored lead at 96
    (results_dir / "modela_test.csv").write_text(CSV_HEADER + "".join([
        "modela,wind_speed,24,1.5,0.9,10,abc,test\n",
        "modela,wind_speed,48,,,0,abc,test\n",
        "modela,wind_speed,72,2.0,0.8,10,abc,test\n",
        "modela,wind_speed,96,3.0,0.7,0,abc,test\n",   # n_inits=0: not real
        "modela,u10,72,1.1,0.85,10,abc,test\n",
    ]))
    # a competitor: the viewer id has no CSV of its own, the _2020 stem does
    (results_dir / "graphcast_2020_test.csv").write_text(
        CSV_HEADER + "graphcast_2020,wind_speed,72,0.85,0.95,10,abc,test\n")
    (results_dir / "sharpness.csv").write_text(
        "model,lead_h,n_inits,ws_rmse,ws_var_ratio,ws_spec_ratio,ws_p95_rmse,"
        "ws_p95_bias,cf_bias,cf_rmse,z500_var_ratio,z500_spec_ratio\n"
        "graphcast_2020,120,10,1.2,0.9,0.7734,2.0,-1.5,0.01,0.134,0.9,0.8\n"
        "avg4 (mean of 4),120,10,1.1,0.9,0.8,1.9,-1.2,0.01,0.12,0.9,0.8\n")


def export(tmp_path, monkeypatch, source_ids):
    write_results(tmp_path / "results")
    docs = tmp_path / "docs"
    (docs / "data").mkdir(parents=True)
    (docs / "data" / "manifest.json").write_text(json.dumps(
        {"sources": [{"id": s} for s in source_ids]}))
    monkeypatch.setattr(bs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(bs, "DOCS", docs)
    out = bs.export_metrics()
    on_disk = json.loads((docs / "data" / "metrics.json").read_text())
    assert on_disk == out
    return out


def test_arrays_align_to_lead_hours_with_nulls(tmp_path, monkeypatch):
    out = export(tmp_path, monkeypatch, ["modela"])
    L = out["lead_hours"]
    assert L == sorted(L) and L[0] == 6 and L[-1] == 120
    ws = out["models"]["modela"]["wind_speed"]
    assert len(ws["rmse"]) == len(L) == len(ws["acc"])
    assert ws["rmse"][L.index(24)] == 1.5
    assert ws["rmse"][L.index(72)] == 2.0
    assert ws["rmse"][L.index(48)] is None          # NaN row dropped
    assert ws["rmse"][L.index(96)] is None          # n_inits=0 dropped
    assert out["models"]["modela"]["u10"]["rmse"][L.index(72)] == 1.1


def test_competitor_resolves_through_2020_stem(tmp_path, monkeypatch):
    out = export(tmp_path, monkeypatch, ["graphcast"])
    assert out["models"]["graphcast"]["csv"] == "graphcast_2020"
    assert out["models"]["graphcast"]["wind_speed"]["rmse"][
        out["lead_hours"].index(72)] == 0.85


def test_sources_without_csv_are_skipped(tmp_path, monkeypatch):
    out = export(tmp_path, monkeypatch, ["modela", "aifs_live", "era5"])
    assert "aifs_live" not in out["models"]         # live: no hindcast scores
    assert "era5" not in out["models"]              # truth: nothing to score


def test_sharpness_keys_map_to_viewer_ids(tmp_path, monkeypatch):
    out = export(tmp_path, monkeypatch, ["graphcast"])
    assert out["sharpness"]["graphcast"]["120"]["ws_spec_ratio"] == 0.7734
    assert "avg4" in out["sharpness"]               # display-name unmangled


def test_provenance_present(tmp_path, monkeypatch):
    out = export(tmp_path, monkeypatch, ["modela"])
    assert "2020" in out["provenance"] and "NOT verified" in out["provenance"]

"""Export pressure-level wind fields for the viewer, streamed from WB2.

The full atmosphere for the hindcast side: u/v at 850/700/500/250 hPa for
ERA5 truth, the frontier-model archives, and persistence -- read directly
from the public WeatherBench-2 zarrs at 64x32, so this needs neither torch,
nor the ERA5 cache, nor a checkpoint. Run it from .venv-dyn (zarr>=3 +
xarray + gcsfs).

Our own checkpoints predict 850 hPa wind too, but producing their fields
means model inference in the training environment: `export_wind.py --level
850` does that where torch and the ERA5 cache exist.

    .venv-dyn/bin/python scripts/export_levels.py
    .venv-dyn/bin/python scripts/export_levels.py --levels 850 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

OUT_DIR = Path("docs/data")
ERA5_URL = ("gs://weatherbench2/datasets/era5/"
            "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")
# forecast archives: (viewer base id, label, url). Each carries a different
# subset of levels; whatever a store lacks is simply skipped for it.
ARCHIVES = [
    ("graphcast", "GraphCast (DeepMind)",
     "gs://weatherbench2/datasets/graphcast/2020/"
     "date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr"),
    ("gencast_mean", "GenCast ensemble mean (DeepMind)",
     "gs://weatherbench2/datasets/gencast/2020-64x32_equiangular_conservative_mean.zarr"),
    ("pangu", "Pangu-Weather (Huawei)",
     "gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr"),
    ("hres", "ECMWF HRES",
     "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr"),
    ("fuxi", "FuXi",
     "gs://weatherbench2/datasets/fuxi/2020-64x32_equiangular_conservative.zarr"),
]
DEFAULT_LEVELS = [925, 850, 700, 500, 300, 250, 200]
ANON = {"token": "anon"}


def velocity_records(u: np.ndarray, v: np.ndarray, lat: np.ndarray,
                     lon: np.ndarray, ref_time: str, forecast_hours: int) -> list:
    """Same grib2json payload as export_wind.velocity_records (kept in copy:
    that module imports torch at the top, which this environment lacks)."""
    order = np.argsort(-lat)  # north first
    u_n, v_n = u[order], v[order]
    lat_sorted = lat[order]
    dy = float(abs(lat_sorted[0] - lat_sorted[1]))
    dx = float(abs(lon[1] - lon[0]))

    def header(param_number: int) -> dict:
        return {
            "parameterUnit": "m.s-1",
            "parameterNumber": param_number,
            "parameterNumberName": "U-component_of_wind" if param_number == 2
                                   else "V-component_of_wind",
            "parameterCategory": 2,
            "nx": len(lon), "ny": len(lat),
            "lo1": float(lon[0]), "la1": float(lat_sorted[0]),
            "lo2": float(lon[-1]), "la2": float(lat_sorted[-1]),
            "dx": dx, "dy": dy,
            "refTime": ref_time, "forecastTime": forecast_hours,
        }

    return [
        {"header": header(2), "data": [round(float(x), 3) for x in u_n.ravel()]},
        {"header": header(3), "data": [round(float(x), 3) for x in v_n.ravel()]},
    ]


def write_payload(out_dir: Path, sid: str, init_s: str, lead: int,
                  u, v, lat, lon) -> None:
    payload = velocity_records(np.asarray(u), np.asarray(v), lat, lon,
                               f"{init_s}:00:00Z", lead)
    (out_dir / f"{sid}_{init_s}_{lead:03d}.json").write_text(
        json.dumps(payload, separators=(",", ":")))


def upsert(manifest: dict, entry: dict) -> None:
    manifest["sources"] = [s for s in manifest["sources"]
                           if s["id"] != entry["id"]] + [entry]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inits", nargs="+",
                   default=["2020-01-15T00", "2020-07-15T00", "2020-10-28T00"])
    p.add_argument("--leads", nargs="+", type=int,
                   default=[0, 24, 48, 72, 96, 120])
    p.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    p.add_argument("--skip-era5", action="store_true",
                   help="only the forecast archives (reuses committed truth files)")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else \
        {"inits": args.inits, "leads": args.leads, "sources": []}

    inits64 = {s: np.datetime64(s, "h") for s in args.inits}

    print("opening ERA5 ...")
    era5 = xr.open_zarr(ERA5_URL, storage_options=ANON, chunks=None)
    lat = era5.latitude.values
    lon = era5.longitude.values

    # ERA5 truth + persistence at each level. Lead 0 of every forecast source
    # is also ERA5's analysis, matching how export_wind.py treats the surface.
    analyses: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for lev in args.levels:
        if lev not in era5.level.values:
            print(f"ERA5 lacks {lev} hPa; skipping the level entirely")
            continue
        if args.skip_era5:
            for init_s, t0 in inits64.items():
                da = era5[["u_component_of_wind", "v_component_of_wind"]] \
                    .sel(time=t0, level=lev)
                analyses[(lev, init_s)] = (da.u_component_of_wind.values,
                                           da.v_component_of_wind.values)
            continue
        for base, kind in (("era5", "truth"), ("persistence", "persistence")):
            sid = f"{base}{lev}"
            for init_s, t0 in inits64.items():
                for lead in args.leads:
                    # persistence IS the analysis held constant at every lead
                    valid = t0 + np.timedelta64(lead if base == "era5" else 0, "h")
                    if (lev, init_s) in analyses and valid == t0:
                        u, v = analyses[(lev, init_s)]
                    else:
                        da = era5[["u_component_of_wind", "v_component_of_wind"]] \
                            .sel(time=valid, level=lev)
                        u = da.u_component_of_wind.values
                        v = da.v_component_of_wind.values
                    if base == "era5" and lead == 0:
                        analyses[(lev, init_s)] = (u, v)
                    write_payload(out_dir, sid, init_s, lead, u, v, lat, lon)
            upsert(man, {
                "id": sid,
                "label": "ERA5 (truth)" if base == "era5" else "persistence",
                "kind": kind, "base": base, "level": f"{lev}hPa",
                "inits": args.inits, "leads": args.leads,
            })
            print(f"exported {sid}")

    for base, label, url in ARCHIVES:
        print(f"opening {base} ...")
        ds = xr.open_zarr(url, storage_options=ANON, chunks=None)
        have = [lev for lev in args.levels if lev in ds.level.values]
        if not have:
            print(f"  {base}: none of {args.levels} present; skipped")
            continue
        for lev in have:
            sid = f"{base}{lev}"
            wrote_leads: set[int] = set()
            for init_s, t0 in inits64.items():
                for lead in args.leads:
                    if lead == 0:
                        if (lev, init_s) not in analyses:
                            continue
                        u, v = analyses[(lev, init_s)]
                    else:
                        # WB2 stores prediction_timedelta as either a real
                        # timedelta or plain int64 HOURS, per archive
                        td = (lead if np.issubdtype(ds.prediction_timedelta.dtype,
                                                    np.integer)
                              else np.timedelta64(lead, "h"))
                        try:
                            da = ds[["u_component_of_wind",
                                     "v_component_of_wind"]].sel(
                                time=t0, prediction_timedelta=td, level=lev)
                        except KeyError:
                            continue      # init or lead not in this archive
                        u = da.u_component_of_wind.values
                        v = da.v_component_of_wind.values
                        if not (np.isfinite(u).all() and np.isfinite(v).all()):
                            continue
                    write_payload(out_dir, sid, init_s, lead, u, v, lat, lon)
                    wrote_leads.add(lead)
            if wrote_leads:
                upsert(man, {
                    "id": sid, "label": label, "kind": "competitor",
                    "base": base, "level": f"{lev}hPa",
                    "inits": args.inits, "leads": sorted(wrote_leads),
                })
                print(f"exported {sid} leads={sorted(wrote_leads)}")

    man_path.write_text(json.dumps(man, indent=2))
    total = sum(f.stat().st_size for f in out_dir.glob("*.json"))
    print(f"manifest updated; {out_dir} now {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()

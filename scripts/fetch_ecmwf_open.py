"""Live pressure-level wind from ECMWF open data (CC BY 4.0, attribute ECMWF).

No dynamical.org store carries wind above 100 m, but ECMWF's open-data
service publishes both IFS (physics) and AIFS (ML) with u/v on pressure
levels at 0.25 degrees. This fetches 850/700/500/250 hPa for both models --
giving the viewer a live full column AND a two-model upper-air blend with
spread -- plus IFS 10 m wind, which makes IFS a fourth member of the global
surface blend.

Everything is interpolated onto the EXACT 2-degree grid the dynamical.org
exports use (la1 89, dx 2), because blend_live.py refuses grids that differ
by a millidegree, and it is right to.

    .venv-dyn/bin/python scripts/fetch_ecmwf_open.py --model aifs
    .venv-dyn/bin/python scripts/fetch_ecmwf_open.py --model ifs
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from ecmwf.opendata import Client

OUT_DIR = Path("docs/data")
LEVELS = [850, 700, 500, 250]
LEADS = list(range(0, 121, 6))          # upper air is 6-hourly by choice
# the grid every global live export shares -- cell CENTERS at 89.125/0.875,
# exactly the dynamical.org coarsen-8 grid WeatherNext is hard-pinned to.
# Header equality is load-bearing: blend_live refuses 89.0 vs 89.125, and the
# first run on a rounder-looking grid proved it right by silently keeping the
# blend a cycle stale.
TARGET_LAT = np.arange(89.125, -90.0, -2.0)
TARGET_LON = np.arange(0.875, 360.0, 2.0)

MODELS = {
    # surface too: ECMWF publishes AIFS hours before dynamical.org's mirror
    # re-serves it, and a member stuck on the previous cycle drops out of the
    # blend. Run AFTER fetch_dynamical so the fresher write wins.
    "aifs": {"client_model": "aifs-single", "id": "aifs_live",
             "label": "ECMWF AIFS (operational ML) — live",
             "surface": True},
    "ifs": {"client_model": "ifs", "id": "ifs_live",
            "label": "ECMWF IFS (operational physics) — live",
            "surface": True},
}


def velocity_records(u, v, lat, lon, ref_time: str, forecast_hours: int) -> list:
    """grib2json-shaped payload; a copy, like the other fetchers carry, so
    each script stays runnable in its own minimal environment."""
    def header(param_number: int) -> dict:
        return {
            "parameterUnit": "m.s-1",
            "parameterNumber": param_number,
            "parameterNumberName": "U-component_of_wind" if param_number == 2
                                   else "V-component_of_wind",
            "parameterCategory": 2,
            "nx": len(lon), "ny": len(lat),
            "lo1": float(lon[0]), "la1": float(lat[0]),
            "lo2": float(lon[-1]), "la2": float(lat[-1]),
            "dx": float(abs(lon[1] - lon[0])), "dy": float(abs(lat[0] - lat[1])),
            "refTime": ref_time, "forecastTime": forecast_hours,
        }
    return [
        {"header": header(2), "data": [round(float(x), 1) for x in np.asarray(u).ravel()]},
        {"header": header(3), "data": [round(float(x), 1) for x in np.asarray(v).ravel()]},
    ]


def to_target(ds: xr.Dataset) -> xr.Dataset:
    """0.25-degree ECMWF grid (lon -180..180) -> our exact 2-degree grid."""
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
    return ds.interp(latitude=TARGET_LAT, longitude=TARGET_LON, method="linear")


def upsert(man: dict, entry: dict) -> None:
    man["sources"] = [s for s in man["sources"] if s["id"] != entry["id"]] + [entry]


def register(man: dict, sid: str, label: str, level: str, base: str,
             leads: list[int], init_iso: str) -> None:
    upsert(man, {"id": sid, "label": label, "kind": "live", "level": level,
                 "base": base, "inits": ["latest"], "leads": leads,
                 "init_time": init_iso})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="aifs", choices=sorted(MODELS))
    p.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    p.add_argument("--leads", nargs="+", type=int, default=LEADS)
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--surface-only", action="store_true")
    args = p.parse_args()

    spec = MODELS[args.model]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(source="ecmwf", model=spec["client_model"])

    if args.surface_only:
        args.levels = []
    with tempfile.TemporaryDirectory() as tmp:
        if args.levels:
            grib = Path(tmp) / "levels.grib2"
            print(f"retrieving {args.model} u/v at {args.levels} hPa, "
                  f"{len(args.leads)} steps ...")
            res = client.retrieve(type="fc", stream="oper", param=["u", "v"],
                                  levelist=args.levels, step=args.leads,
                                  target=str(grib))
        else:
            res = client.retrieve(type="fc", stream="oper", param=["10u", "10v"],
                                  step=[0], target=str(Path(tmp) / "probe.grib2"))
        init = res.datetime
        init_iso = init.strftime("%Y-%m-%dT%H:00:00Z")
        print(f"init {init_iso}")
        steps = args.leads
        if args.levels:
            ds = to_target(xr.open_dataset(grib, engine="cfgrib",
                                           backend_kwargs={"indexpath": ""}))
            steps = (ds.step.values.astype("timedelta64[h]").astype(int)
                     if "step" in ds.dims else [0])
        for lev in args.levels:
            sid = spec["id"].replace("_live", f"{lev}_live")
            wrote = []
            for i, h in enumerate(steps):
                sel = ds.sel(isobaricInhPa=lev).isel(step=i)
                payload = velocity_records(sel.u.values, sel.v.values,
                                           TARGET_LAT, TARGET_LON, init_iso, int(h))
                (out_dir / f"{sid}_latest_{int(h):03d}.json").write_text(
                    json.dumps(payload, separators=(",", ":")))
                wrote.append(int(h))
            print(f"  {sid}: leads {wrote[0]}..{wrote[-1]} ({len(wrote)})")

        if spec["surface"]:
            grib_s = Path(tmp) / "surface.grib2"
            client.retrieve(type="fc", stream="oper", param=["10u", "10v"],
                            step=args.leads, target=str(grib_s))
            ss = to_target(xr.open_dataset(grib_s, engine="cfgrib",
                                           backend_kwargs={"indexpath": ""}))
            steps = (ss.step.values.astype("timedelta64[h]").astype(int)
                     if "step" in ss.dims else [0])
            for i, h in enumerate(steps):
                sel = ss.isel(step=i)
                payload = velocity_records(sel.u10.values, sel.v10.values,
                                           TARGET_LAT, TARGET_LON, init_iso, int(h))
                (out_dir / f"{spec['id']}_latest_{int(h):03d}.json").write_text(
                    json.dumps(payload, separators=(",", ":")))
            print(f"  {spec['id']}: 10 m surface, {len(steps)} leads")

    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else \
        {"inits": ["latest"], "leads": list(steps), "sources": []}
    leads_list = [int(h) for h in steps]
    for lev in args.levels:
        register(man, spec["id"].replace("_live", f"{lev}_live"),
                 spec["label"].replace(" — live", f" — live, {lev} hPa"),
                 f"{lev}hPa", spec["id"], leads_list, init_iso)
    if spec["surface"]:
        register(man, spec["id"], spec["label"], "10m", spec["id"], leads_list,
                 init_iso)
        # base entries carry no separate base key
        for s in man["sources"]:
            if s["id"] == spec["id"]:
                s.pop("base", None)
    man_path.write_text(json.dumps(man, indent=2))
    print(f"registered in {man_path}")


if __name__ == "__main__":
    main()

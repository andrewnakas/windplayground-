"""Add live operational forecasts from dynamical.org to the wind viewer.

dynamical.org republishes weather archives as cloud-optimized Zarr v3, readable
over plain HTTPS with no credentials. We use ECMWF **AIFS** by default: it is
the cheapest to fetch (one chunk holds a whole init's 61 lead times) and it is
the operational ML forecast system discussed in REPORT.md, so it belongs
alongside the research models in the viewer.

Requires zarr>=3, which conflicts with the training env's pinned zarr<3 --
hence a separate interpreter:

    uv venv .venv-dyn
    uv pip install --python .venv-dyn/bin/python "zarr>=3" "xarray>=2025.1" numpy aiohttp
    .venv-dyn/bin/python scripts/fetch_dynamical.py

Note on cost: chunks are [1 init, 61 leads, 721 lat, 1440 lon] = 254 MB raw per
variable, so a run downloads a few hundred MB and yields every lead time.
Fields are coarsened before export to keep the viewer payload small.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

DATASETS = {
    "aifs": {
        "url": "https://data.dynamical.org/ecmwf/aifs-single/forecast/latest.zarr",
        "label": "ECMWF AIFS (operational ML) — live",
        "id": "aifs_live",
    },
    "gfs": {
        "url": "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        "label": "NOAA GFS (operational physics) — live",
        "id": "gfs_live",
    },
}
OUT_DIR = Path("viewer/data")


def velocity_records(u, v, lat, lon, ref_time: str, forecast_hours: int) -> list:
    """Same grib2json layout as scripts/export_wind.py: north-first, row-major."""
    if lat[0] < lat[-1]:                      # ensure north -> south
        lat, u, v = lat[::-1], u[::-1], v[::-1]
    lon = np.asarray(lon) % 360.0
    order = np.argsort(lon)                   # ensure 0 -> 360 east
    lon, u, v = lon[order], u[:, order], v[:, order]

    def header(param_number: int) -> dict:
        return {
            "parameterUnit": "m.s-1",
            "parameterNumber": param_number,
            "parameterNumberName": "U-component_of_wind" if param_number == 2
                                   else "V-component_of_wind",
            "parameterCategory": 2,
            "nx": int(len(lon)), "ny": int(len(lat)),
            "lo1": float(lon[0]), "la1": float(lat[0]),
            "lo2": float(lon[-1]), "la2": float(lat[-1]),
            "dx": float(abs(lon[1] - lon[0])), "dy": float(abs(lat[0] - lat[1])),
            "refTime": ref_time, "forecastTime": forecast_hours,
        }

    return [
        {"header": header(2), "data": [round(float(x), 3) for x in np.nan_to_num(u).ravel()]},
        {"header": header(3), "data": [round(float(x), 3) for x in np.nan_to_num(v).ravel()]},
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="aifs", choices=sorted(DATASETS))
    p.add_argument("--leads", nargs="+", type=int, default=[0, 24, 48, 72, 96, 120])
    p.add_argument("--coarsen", type=int, default=4,
                   help="spatial coarsening factor (4 -> 1 degree from 0.25 degree)")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    print(f"opening {spec['url']} ...")
    ds = xr.open_zarr(spec["url"], chunks=None, consolidated=True)

    init = ds.init_time.values[-1]            # most recent run
    print(f"latest init_time: {init}")
    lead_td = np.array([np.timedelta64(h, "h") for h in args.leads])
    available = set(ds.lead_time.values.astype("timedelta64[h]").astype(int))
    keep_h = [h for h in args.leads if h in available]
    if not keep_h:
        raise SystemExit(f"none of {args.leads} are available; first few: "
                         f"{sorted(available)[:8]}")
    lead_td = np.array([np.timedelta64(h, "h") for h in keep_h])

    sub = ds[["wind_u_10m", "wind_v_10m"]].sel(init_time=init, lead_time=lead_td)
    print("downloading (one chunk covers every lead time) ...")
    sub = sub.load()

    if args.coarsen > 1:
        sub = sub.coarsen(latitude=args.coarsen, longitude=args.coarsen,
                          boundary="trim").mean()

    lat = sub.latitude.values
    lon = sub.longitude.values
    ref = str(np.datetime64(init, "h"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_label = ref.replace(":00", "")       # matches the viewer's naming
    for i, h in enumerate(keep_h):
        u = sub.wind_u_10m.isel(lead_time=i).values
        v = sub.wind_v_10m.isel(lead_time=i).values
        payload = velocity_records(u, v, lat, lon, ref + ":00:00Z", h)
        path = out_dir / f"{spec['id']}_{init_label}_{h:03d}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"  +{h:3d} h -> {path.name} ({path.stat().st_size/1e6:.1f} MB)")

    # register the live source in the viewer manifest
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else \
        {"inits": [], "leads": keep_h, "sources": []}
    if init_label not in man["inits"]:
        man["inits"].append(init_label)
    man["leads"] = sorted(set(man["leads"]) | set(keep_h))
    man["sources"] = [s for s in man["sources"] if s["id"] != spec["id"]]
    man["sources"].append({"id": spec["id"], "label": spec["label"], "kind": "live",
                           "inits": [init_label], "leads": keep_h})
    man_path.write_text(json.dumps(man, indent=2))
    print(f"\nregistered {spec['id']} for init {init_label} in {man_path}")


if __name__ == "__main__":
    main()

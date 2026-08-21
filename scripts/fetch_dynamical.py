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
    # Probed 2026-08-10, in answer to "can we add GenCast / GraphCast / Pangu /
    # FuXi live?". Short answer: no, they do not exist as live public feeds.
    # ECMWF's open data serves only aifs-ens, aifs-single and ifs; dynamical.org's
    # catalog carries no ML research models at all. What WeatherBench-2 ships for
    # those four is 2020 (and 2018 GraphCast) *archives*. Running them live would
    # mean self-hosting inference against real-time initial conditions -- open
    # weights exist for GraphCast, Pangu and GenCast, but GenCast is a diffusion
    # ensemble needing TPU-scale compute per forecast.
    #
    # The one live product that would genuinely improve this blend is ECMWF
    # **IFS ENS** (51 members, the physics ensemble GenCast is benchmarked
    # against) at ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr -- it is
    # reachable and carries wind_u_10m/wind_v_10m. It is left out on measured
    # cost, not preference. Chunk layout decides everything here:
    #
    #     aifs-single  (1, 61, 241, 240)      14 MB x   18 =  0.25 GB /var/init
    #     gfs          (1, 105, 121, 121)      6 MB x  144 =  0.89 GB /var/init
    #     ifs-ens      (1, 85, 51, 32, 32)    18 MB x 1035 = 18.38 GB /var/init
    #
    # IFS ENS tiles space at 32x32 while packing all 85 leads and all 51 members
    # into every tile, so a global field costs the entire ensemble even though we
    # only want the mean at six leads: ~37 GB for u and v against this container's
    # ~3 GB of disk. AIFS ENS has a catalog page but no zarr at any path probed.
    #
    # Earlier probe (2026-08-06): ecmwf/ifs, noaa/hrrr and noaa/gefs/forecast 404.
    "gefs": {
        "url": "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr",
        "label": "NOAA GEFS 35-day ensemble — live",
        "id": "gefs_live",
        # an ensemble: collapse members before export rather than picking one
        "ensemble_mean": True,
        # NOT in the automated refresh set. Tried it: a 35-day ensemble is a
        # far larger download than the 0-120h products and did not finish in a
        # sensible window, and its long leads are not what this viewer shows.
        # Reachable on demand with --dataset gefs; left out of the cron.
        "skip_scheduled": True,
    },
}
OUT_DIR = Path("docs/data")

# Full 6-hourly ladder to +120 h. AIFS is 6-hourly and GFS hourly upstream, so
# every rung exists for both; 21 leads at 2 degrees is ~3 MB per model per
# refresh, the price of a scrubber and meteogram that move in 6 h steps
# instead of 24. WeatherNext is 6-hourly too (from +6), so the blend keeps
# every rung as well.
DEFAULT_LEADS = list(range(0, 121, 6))


def select_leads(requested: list[int], available: set[int]) -> list[int]:
    """The requested leads the store actually carries, in request order."""
    keep = [h for h in requested if h in available]
    if not keep:
        raise SystemExit(f"none of {requested} are available; first few: "
                         f"{sorted(available)[:8]}")
    return keep


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
            "nx": len(lon), "ny": len(lat),
            "lo1": float(lon[0]), "la1": float(lat[0]),
            "lo2": float(lon[-1]), "la2": float(lat[-1]),
            "dx": float(abs(lon[1] - lon[0])), "dy": float(abs(lat[0] - lat[1])),
            "refTime": ref_time, "forecastTime": forecast_hours,
        }

    return [
        {"header": header(2), "data": [round(float(x), 1) for x in np.nan_to_num(u).ravel()]},
        {"header": header(3), "data": [round(float(x), 1) for x in np.nan_to_num(v).ravel()]},
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="aifs", choices=sorted(DATASETS))
    p.add_argument("--leads", nargs="+", type=int, default=DEFAULT_LEADS)
    # 8 -> a 2-degree grid (~0.2 MB/lead). This is a git-churn decision, not a
    # visual one: these files are committed on every 6-hourly refresh, and at
    # the previous 1 degree they were 0.8 MB each, ~10 MB per commit. 2 degrees
    # is still 2.8x finer than our own models' 5.625 degree grid and the
    # particle animation is indistinguishable below zoom 6. Wind is rounded to
    # 0.1 m/s for the same reason -- three decimals of a visualised field is
    # bytes spent on nothing.
    # Hub height: both AIFS and GFS also publish 100 m wind (GFS's nearest is
    # wind_{u,v}_100m too). Gusts do NOT exist in either store -- probed
    # 2026-08-20 -- so 100 m is the one richer wind variable on offer, and it
    # is the one the capacity-factor work in RESULTS.md actually wants.
    p.add_argument("--level", default="10m", choices=("10m", "100m"))
    p.add_argument("--coarsen", type=int, default=8,
                   help="spatial coarsening factor (4 -> 1 degree from 0.25 degree)")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    print(f"opening {spec['url']} ...")
    ds = xr.open_zarr(spec["url"], chunks=None, consolidated=True)

    init = ds.init_time.values[-1]            # most recent run
    print(f"latest init_time: {init}")
    available = set(ds.lead_time.values.astype("timedelta64[h]").astype(int))
    keep_h = select_leads(args.leads, available)
    lead_td = np.array([np.timedelta64(h, "h") for h in keep_h])

    uvar, vvar = f"wind_u_{args.level}", f"wind_v_{args.level}"
    if uvar not in ds:
        raise SystemExit(f"{args.dataset} has no {uvar}; variables: "
                         f"{sorted(ds.data_vars)[:12]} ...")
    sub = ds[[uvar, vvar]].sel(init_time=init, lead_time=lead_td)
    print("downloading (one chunk covers every lead time) ...")
    sub = sub.load()

    # GEFS is an ensemble; average the members so the viewer gets one field.
    # Doing this before coarsening keeps it a plain mean over realizations.
    if spec.get("ensemble_mean"):
        member_dims = [d for d in ("ensemble_member", "realization", "number")
                       if d in sub.dims]
        if member_dims:
            print(f"averaging ensemble over {member_dims[0]} "
                  f"({sub.sizes[member_dims[0]]} members)")
            sub = sub.mean(dim=member_dims[0], keep_attrs=True)

    if args.coarsen > 1:
        sub = sub.coarsen(latitude=args.coarsen, longitude=args.coarsen,
                          boundary="trim").mean()

    lat = sub.latitude.values
    lon = sub.longitude.values
    ref = str(np.datetime64(init, "h"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # STABLE filename, with the real init time carried in the manifest instead.
    # Timestamped paths meant every refresh added files and never removed any,
    # so an automated 6-hourly job would grow the repo without bound. "latest"
    # overwrites in place and keeps each commit a bounded diff.
    init_label = "latest"
    actual_init = ref.replace(":00", "")
    # 100 m fields live under their own source id (aifs100_live) so each level
    # blends and displays independently; the viewer pairs them by this naming.
    source_id = spec["id"] if args.level == "10m" else \
        spec["id"].replace("_live", "100_live")
    label = spec["label"] if args.level == "10m" else \
        spec["label"].replace(" — live", " — live, 100 m")
    for i, h in enumerate(keep_h):
        u = sub[uvar].isel(lead_time=i).values
        v = sub[vvar].isel(lead_time=i).values
        payload = velocity_records(u, v, lat, lon, ref + ":00:00Z", h)
        path = out_dir / f"{source_id}_{init_label}_{h:03d}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"  +{h:3d} h -> {path.name} ({path.stat().st_size/1e6:.1f} MB)")

    # register the live source in the viewer manifest
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else \
        {"inits": [], "leads": keep_h, "sources": []}
    if init_label not in man["inits"]:
        man["inits"].append(init_label)
    man["leads"] = sorted(set(man["leads"]) | set(keep_h))
    man["sources"] = [s for s in man["sources"] if s["id"] != source_id]
    man["sources"].append({"id": source_id, "label": label, "kind": "live",
                           "level": args.level,
                           "inits": [init_label], "leads": keep_h,
                           # what the viewer shows as "init 06Z, 3 h ago"; the
                           # filename no longer carries it
                           "init_time": actual_init + ":00:00Z"})
    man_path.write_text(json.dumps(man, indent=2))
    print(f"\nregistered {source_id} for init {init_label} in {man_path}")


if __name__ == "__main__":
    main()

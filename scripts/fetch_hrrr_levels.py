"""HRRR pressure-level wind from the NOAA AWS archive (anonymous, free).

dynamical.org's HRRR zarr stops at the surface, but the native wrfprs files
carry the full column. Each GRIB2 ships with an .idx sidecar mapping every
field to a byte range, so this downloads ONLY the u/v messages at the wanted
levels -- a few hundred KB per field against the ~400 MB file -- decodes them
with eccodes, and bins the Lambert grid onto the same regular lat/lon box the
surface HRRR export uses.

    .venv-dyn/bin/python scripts/fetch_hrrr_levels.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_dynamical import bin_regrid   # noqa: E402  (same env, same grid rules)

OUT_DIR = Path("docs/data")
BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
LEVELS = [925, 850, 700, 500, 300, 250, 200]
LEADS = list(range(0, 49, 6))            # upper air 6-hourly; surface is hourly
BASE_ID = "hrrr_live"
LABEL = "NOAA HRRR (3 km CONUS) — live"


def latest_cycle(max_back: int = 12) -> tuple[str, int]:
    """Newest init whose +48 h wrfprs file exists (the run is complete)."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for back in range(1, max_back + 1):
        t = now - timedelta(hours=back)
        day, hh = t.strftime("%Y%m%d"), t.hour
        url = f"{BUCKET}/hrrr.{day}/conus/hrrr.t{hh:02d}z.wrfprsf48.grib2.idx"
        try:
            with urllib.request.urlopen(url, timeout=20):
                return day, hh
        except Exception:
            continue
    raise SystemExit("no complete HRRR cycle found in the last 12 hours")


def field_ranges(idx_text: str, wanted: set[tuple[str, int]]) -> dict:
    """(var, level) -> (start, end) byte range from the .idx sidecar."""
    lines = idx_text.strip().split("\n")
    starts = [int(ln.split(":")[1]) for ln in lines]
    out = {}
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        var, level = parts[3], parts[4]
        for v, lev in wanted:
            if var == v and level == f"{lev} mb":
                end = starts[i + 1] - 1 if i + 1 < len(starts) else ""
                out[(v, lev)] = (starts[i], end)
    return out


def fetch_ranged(url: str, rng: tuple) -> bytes:
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={rng[0]}-{rng[1]}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    p.add_argument("--leads", nargs="+", type=int, default=LEADS)
    p.add_argument("--regional-res", type=float, default=0.2)
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    day, hh = latest_cycle()
    init_iso = f"{day[:4]}-{day[4:6]}-{day[6:]}T{hh:02d}:00:00Z"
    print(f"HRRR cycle {init_iso}")
    wanted = {(v, lev) for v in ("UGRD", "VGRD") for lev in args.levels}

    # gather every (lead, level) field first, then regrid all at once so the
    # cropped box is identical across leads and levels
    fields: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    lat2d = lon2d = None
    with tempfile.TemporaryDirectory() as tmp:
        for h in args.leads:
            base = f"{BUCKET}/hrrr.{day}/conus/hrrr.t{hh:02d}z.wrfprsf{h:02d}.grib2"
            idx = urllib.request.urlopen(base + ".idx", timeout=30).read().decode()
            ranges = field_ranges(idx, wanted)
            per_level: dict[int, dict[str, np.ndarray]] = {}
            for (var, lev), rng in sorted(ranges.items()):
                blob = fetch_ranged(base, rng)
                gp = Path(tmp) / "msg.grib2"
                gp.write_bytes(blob)
                ds = xr.open_dataset(gp, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
                name = [v for v in ds.data_vars][0]
                per_level.setdefault(lev, {})[var] = ds[name].values
                if lat2d is None:
                    lat2d = ds.latitude.values
                    lon2d = ds.longitude.values
            for lev, uv in per_level.items():
                if "UGRD" in uv and "VGRD" in uv:
                    fields[(h, lev)] = (uv["UGRD"], uv["VGRD"])
            print(f"  +{h:02d} h: {len(per_level)} levels")

    keys = sorted(fields)
    stack = [a for k in keys for a in fields[k]]
    lat, lon, regridded = bin_regrid(lat2d, lon2d, stack, args.regional_res)

    def rec(vals, param):
        return {"header": {
            "parameterUnit": "m.s-1", "parameterNumber": param,
            "parameterNumberName": "U-component_of_wind" if param == 2
                                   else "V-component_of_wind",
            "parameterCategory": 2, "nx": len(lon), "ny": len(lat),
            "lo1": float(lon[0]), "la1": float(lat[0]),
            "lo2": float(lon[-1]), "la2": float(lat[-1]),
            "dx": float(abs(lon[1] - lon[0])), "dy": float(abs(lat[0] - lat[1])),
            "refTime": init_iso, "forecastTime": 0,
        }, "data": [round(float(x), 1) for x in np.asarray(vals).ravel()]}

    wrote: dict[int, list[int]] = {}
    for i, (h, lev) in enumerate(keys):
        u, v = regridded[2 * i], regridded[2 * i + 1]
        payload = [rec(u, 2), rec(v, 3)]
        payload[0]["header"]["forecastTime"] = h
        payload[1]["header"]["forecastTime"] = h
        sid = BASE_ID.replace("_live", f"{lev}_live")
        (out_dir / f"{sid}_latest_{h:03d}.json").write_text(
            json.dumps(payload, separators=(",", ":")))
        wrote.setdefault(lev, []).append(h)

    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else \
        {"inits": ["latest"], "leads": args.leads, "sources": []}
    for lev, leads in sorted(wrote.items()):
        sid = BASE_ID.replace("_live", f"{lev}_live")
        man["sources"] = [s for s in man["sources"] if s["id"] != sid] + [{
            "id": sid, "label": LABEL.replace(" — live", f" — live, {lev} hPa"),
            "kind": "live", "level": f"{lev}hPa", "base": BASE_ID,
            "domain": "conus", "inits": ["latest"], "leads": sorted(leads),
            "init_time": init_iso,
        }]
        print(f"registered {sid} leads={sorted(leads)}")
    man_path.write_text(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()

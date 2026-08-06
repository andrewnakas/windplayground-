"""Build the RT2021 ERA5 training stack. Runs as a Kaggle kernel.

Rasp & Thuerey (2021) feed 38 fields per time step -- geopotential,
temperature, u, v and specific humidity on 7 pressure levels (35), plus 2m
temperature, 6-hourly precipitation and TOA incident solar radiation. Stacked
over t, t-6h and t-12h that is their 114 dynamic channels; land-sea mask,
orography and latitude bring the network input to 117.

This runs on Kaggle rather than in the dev container for a blunt reason: the
container has ~3.8 GB of free disk and the output is ~9 GB. Kaggle also mounts
its own kernel outputs as datasets, so the training kernel reads this directly
with no upload from anyone's laptop.

Output (to /kaggle/working, which becomes a Kaggle Dataset):
    era5_rt2021_<year>.npy   float16 (time, 38, 32, 64)
    stats.json               per-channel mean/std/diff_std over the train years
    meta.json                channel order, levels, provenance
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

# Kaggle's image ships xarray but NOT zarr, so open_zarr dies with
# "unrecognized engine 'zarr'". Install before importing xarray, since the
# backend registry is populated at import time.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=True)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")
HTTPS = WB2.replace("gs://", "https://storage.googleapis.com/")

LEVELS = [50, 250, 500, 600, 700, 850, 925]
LEVEL_VARS = [("geopotential", "z"), ("temperature", "t"),
              ("u_component_of_wind", "u"), ("v_component_of_wind", "v"),
              ("specific_humidity", "q")]
SURFACE = [("2m_temperature", "t2m"), ("total_precipitation_6hr", "tp")]

# Their split. Ours is normally 2020; mixing the two makes any comparison to
# the paper's 314/268 meaningless, so the years are pinned here.
TRAIN, VAL, TEST = (1979, 2015), (2016, 2016), (2017, 2018)

PRECIP_EPS = 1e-3
SOLAR_CONSTANT = 1361.0
OUT = pathlib.Path("/kaggle/working")


def channel_order() -> list[str]:
    """Scored targets first, then the rest -- loss and metrics index by position."""
    names = ["z500", "t850", "t2m"]
    names += [f"{a}{lv}" for _, a in LEVEL_VARS for lv in LEVELS
              if f"{a}{lv}" not in ("z500", "t850")]
    names += ["tp", "tisr"]
    return names


def toa_solar(times: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Analytic TOA insolation (W/m^2). WB2 does not ship this field, and it is
    pure solar geometry, so computing it is exact rather than a substitute."""
    t = np.asarray(times, dtype="datetime64[s]")
    year0 = t.astype("datetime64[Y]").astype("datetime64[s]")
    doy = (t - year0).astype(np.float64) / 86400.0 + 1.0
    day0 = t.astype("datetime64[D]").astype("datetime64[s]")
    hour = (t - day0).astype(np.float64) / 3600.0

    g = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.001480 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dist = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
            + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))

    la = np.deg2rad(lat)[None, :, None]
    ha = np.deg2rad(15.0 * (hour[:, None, None] + lon[None, None, :] / 15.0
                            + eot[:, None, None] / 60.0 - 12.0))
    d = decl[:, None, None]
    cz = np.clip(np.sin(la) * np.sin(d) + np.cos(la) * np.cos(d) * np.cos(ha), 0, None)
    return (SOLAR_CONSTANT * dist[:, None, None] * cz).astype(np.float32)


def build_year(ds: xr.Dataset, year: int, names: list[str]) -> np.ndarray:
    sub = ds.sel(time=str(year))
    lat = sub.latitude.values
    lon = sub.longitude.values
    n_t = sub.sizes["time"]
    out = np.empty((n_t, len(names), len(lat), len(lon)), dtype=np.float16)
    idx = {n: i for i, n in enumerate(names)}

    for var, abbr in LEVEL_VARS:
        block = sub[var].sel(level=LEVELS).transpose(
            "time", "level", "latitude", "longitude").values
        for k, lv in enumerate(LEVELS):
            out[:, idx[f"{abbr}{lv}"]] = block[:, k].astype(np.float16)

    for var, short in SURFACE:
        vals = sub[var].transpose("time", "latitude", "longitude").values
        if short == "tp":
            # log1p(PR/eps) == log(eps+PR)-log(eps) but exactly 0 at PR=0; as a
            # difference of logs in float32 it leaves ~1e-7 of residue there,
            # which defeats the transform's whole purpose (the paper calls it
            # crucial -- untransformed, the net just predicts zeros).
            vals = np.log1p(np.maximum(vals, 0.0) / PRECIP_EPS)
        out[:, idx[short]] = vals.astype(np.float16)

    out[:, idx["tisr"]] = toa_solar(sub.time.values, lat, lon).astype(np.float16)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = channel_order()
    print(f"windml channels={len(names)} (x3 frames + 3 statics = "
          f"{len(names) * 3 + 3} network inputs)", flush=True)

    ds = xr.open_zarr(HTTPS, chunks=None)
    print(f"windml zarr_open=ok years={TRAIN[0]}..{TEST[1]}", flush=True)

    # accumulate train-year statistics incrementally: concatenating 37 years
    # would need far more RAM than a Kaggle kernel has
    tot = sq = dtot = dsq = None
    n = n_d = 0
    tail = None

    for year in range(TRAIN[0], TEST[1] + 1):
        arr = build_year(ds, year, names)
        path = OUT / f"era5_rt2021_{year}.npy"
        np.save(path, arr)
        print(f"windml year={year} shape={arr.shape} "
              f"{path.stat().st_size/1e6:.0f}MB", flush=True)

        if TRAIN[0] <= year <= TRAIN[1]:
            x = arr.astype(np.float64)
            if tot is None:
                tot = np.zeros(len(names)); sq = np.zeros(len(names))
                dtot = np.zeros(len(names)); dsq = np.zeros(len(names))
            tot += x.sum(axis=(0, 2, 3)); sq += np.square(x).sum(axis=(0, 2, 3))
            n += x.shape[0] * x.shape[2] * x.shape[3]
            # carry the last frame across the year boundary so the 6h-difference
            # statistics match processing the whole series at once
            d = np.diff(x if tail is None else np.concatenate([tail[None], x]), axis=0)
            dtot += d.sum(axis=(0, 2, 3)); dsq += np.square(d).sum(axis=(0, 2, 3))
            n_d += d.shape[0] * d.shape[2] * d.shape[3]
            tail = x[-1]
        del arr

    mean = tot / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 0))
    dmean = dtot / n_d
    dstd = np.sqrt(np.maximum(dsq / n_d - dmean**2, 0))
    # precipitation keeps its zero lower bound: scaled by std, mean NOT removed
    mean[names.index("tp")] = 0.0

    (OUT / "stats.json").write_text(json.dumps({
        "channels": names, "mean": mean.tolist(),
        "std": std.tolist(), "diff_std": dstd.tolist()}, indent=2))
    (OUT / "meta.json").write_text(json.dumps({
        "source": WB2, "levels": LEVELS, "channels": names,
        "train_years": TRAIN, "val_years": VAL, "test_years": TEST,
        "precip_transform": "log1p(PR/1e-3), std-scaled, mean not subtracted",
        "tisr": "computed analytically (Spencer 1971), not from the archive",
    }, indent=2))
    print("windml stats=written")
    print("RESULT OK")


if __name__ == "__main__":
    main()

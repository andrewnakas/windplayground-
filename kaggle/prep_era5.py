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

**Arrays are stored NORMALIZED, not raw.** The first version stored physical
units in float16 and produced `inf`: geopotential at 50 hPa is ~202,000 m2/s2
and at 250 hPa ~102,000, both past float16's 65,504 ceiling. Every downstream
tensor went NaN and a 136-minute GPU run was wasted proving it. Normalizing
first also *gains* precision -- fp16 spacing at 54,000 (z500) is 32 m2/s2,
against ~0.0005 sigma once the field is centred and scaled.

The affine actually applied is published as `store_mean`/`store_std`; the exact
train-period `mean`/`std`/`diff_std` are computed alongside it in physical
units, so consumers can recover either.

Output (to /kaggle/working, which becomes a Kaggle Dataset):
    era5_rt2021_<year>.npy   float16 (time, 38, 32, 64), normalized
    stats.json               store affine + exact physical mean/std/diff_std
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

# Pass 1 only has to pick a scale that keeps float16 in range and near unit
# variance -- it is not the published statistic, so three spread-out years at
# stride 8 is ample and costs ~8% of one full pass rather than doubling it.
SCALE_YEARS = (1980, 1997, 2014)
SCALE_STRIDE = 8

PRECIP_EPS = 1e-3
SOLAR_CONSTANT = 1361.0
FP16_MAX = 65504.0
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


def build_block(sub: xr.Dataset, names: list[str]) -> np.ndarray:
    """Assemble one time-slice of the 38-field stack in PHYSICAL units, float32.

    float32 throughout: the cast to float16 happens only after normalization,
    which is the whole point of the two-pass shape.
    """
    lat = sub.latitude.values
    lon = sub.longitude.values
    out = np.empty((sub.sizes["time"], len(names), len(lat), len(lon)),
                   dtype=np.float32)
    idx = {n: i for i, n in enumerate(names)}

    for var, abbr in LEVEL_VARS:
        block = sub[var].sel(level=LEVELS).transpose(
            "time", "level", "latitude", "longitude").values
        for k, lv in enumerate(LEVELS):
            out[:, idx[f"{abbr}{lv}"]] = block[:, k]

    for var, short in SURFACE:
        vals = sub[var].transpose("time", "latitude", "longitude").values
        if short == "tp":
            # log1p(PR/eps) == log(eps+PR)-log(eps) but exactly 0 at PR=0; as a
            # difference of logs in float32 it leaves ~1e-7 of residue there,
            # which defeats the transform's whole purpose (the paper calls it
            # crucial -- untransformed, the net just predicts zeros).
            vals = np.log1p(np.maximum(vals, 0.0) / PRECIP_EPS)
        out[:, idx[short]] = vals

    out[:, idx["tisr"]] = toa_solar(sub.time.values, lat, lon)
    return out


def storage_affine(ds: xr.Dataset, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1: a cheap centre/scale that makes every channel float16-safe."""
    tot = np.zeros(len(names))
    sq = np.zeros(len(names))
    n = 0
    for year in SCALE_YEARS:
        sub = ds.sel(time=str(year)).isel(time=slice(None, None, SCALE_STRIDE))
        x = build_block(sub, names).astype(np.float64)
        tot += x.sum(axis=(0, 2, 3))
        sq += np.square(x).sum(axis=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
        print(f"windml scale_year={year} n={x.shape[0]}", flush=True)

    mean = tot / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 0))
    std[std < 1e-8] = 1.0                       # a constant channel stays itself
    # precipitation keeps its zero lower bound: scaled, never shifted
    mean[names.index("tp")] = 0.0
    return mean, std


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = channel_order()
    print(f"windml channels={len(names)} (x3 frames + 3 statics = "
          f"{len(names) * 3 + 3} network inputs)", flush=True)

    ds = xr.open_zarr(HTTPS, chunks=None)
    print(f"windml zarr_open=ok years={TRAIN[0]}..{TEST[1]}", flush=True)

    store_mean, store_std = storage_affine(ds, names)
    for nm, m, s in zip(names, store_mean, store_std):
        print(f"windml store {nm:>6s} mean={m:12.3f} std={s:11.3f}", flush=True)

    # accumulate exact train-year statistics incrementally: concatenating 37
    # years would need far more RAM than a Kaggle kernel has
    tot = sq = dtot = dsq = None
    n = n_d = 0
    tail = None
    peak = 0.0

    for year in range(TRAIN[0], TEST[1] + 1):
        phys = build_block(ds.sel(time=str(year)), names)
        arr = ((phys - store_mean[None, :, None, None].astype(np.float32))
               / store_std[None, :, None, None].astype(np.float32)).astype(np.float16)

        # Fail at the source. The previous run wrote inf here and the first
        # symptom was `val=nan` two kernels and 136 GPU-minutes downstream.
        if not np.isfinite(arr).all():
            bad = [names[c] for c in range(len(names))
                   if not np.isfinite(arr[:, c]).all()]
            raise SystemExit(f"RESULT FAIL non-finite after float16 cast in "
                             f"year {year}: {bad}")
        peak = max(peak, float(np.abs(arr.astype(np.float32)).max()))

        path = OUT / f"era5_rt2021_{year}.npy"
        np.save(path, arr)
        print(f"windml year={year} shape={arr.shape} absmax={peak:.1f} "
              f"{path.stat().st_size/1e6:.0f}MB", flush=True)

        if TRAIN[0] <= year <= TRAIN[1]:
            x = phys.astype(np.float64)
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
            del x, d
        del phys, arr

    mean = tot / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 0))
    dmean = dtot / n_d
    dstd = np.sqrt(np.maximum(dsq / n_d - dmean**2, 0))
    # precipitation keeps its zero lower bound: scaled by std, mean NOT removed
    mean[names.index("tp")] = 0.0

    (OUT / "stats.json").write_text(json.dumps({
        "channels": names,
        "store_mean": store_mean.tolist(), "store_std": store_std.tolist(),
        "mean": mean.tolist(), "std": std.tolist(), "diff_std": dstd.tolist(),
        "stored_normalized": True}, indent=2))
    (OUT / "meta.json").write_text(json.dumps({
        "source": WB2, "levels": LEVELS, "channels": names,
        "train_years": TRAIN, "val_years": VAL, "test_years": TEST,
        "stored_normalized": True,
        "storage": "arr = (physical - store_mean) / store_std, then float16",
        "storage_scale_years": list(SCALE_YEARS),
        "precip_transform": "log1p(PR/1e-3), std-scaled, mean not subtracted",
        "tisr": "computed analytically (Spencer 1971), not from the archive",
    }, indent=2))
    print(f"windml stats=written absmax={peak:.1f} (float16 max {FP16_MAX:.0f})")
    print("RESULT OK")


if __name__ == "__main__":
    main()

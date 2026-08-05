"""Materialize a local per-year cache of the WB2 ERA5 64x32 dataset.

Each year is saved as artifacts/data/era5_64x32/<year>.npy with shape
(n_time, 8, 32, 64) float32, channels in config.CHANNELS order. Statics are
saved once to statics.npz. Downloads are per-year and skipped when the file
already exists, so an interrupted run resumes where it left off.
"""
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import xarray as xr

from windml.config import STATIC_VARIABLES, DataConfig, active_variables


def open_wb2(url: str) -> xr.Dataset:
    return xr.open_zarr(url, storage_options={"token": "anon"})


def _cache_root(cfg: DataConfig) -> Path:
    suffix = "" if cfg.variable_set == "core" else f"_{cfg.variable_set}"
    return Path(cfg.cache_dir) / f"era5_{cfg.grid}{suffix}"


def year_path(cfg: DataConfig, year: int) -> Path:
    return _cache_root(cfg) / f"{year}.npy"


def statics_path(cfg: DataConfig) -> Path:
    return _cache_root(cfg) / "statics.npz"


def _fetch_year(ds: xr.Dataset, year: int, variables, retries: int = 4) -> np.ndarray:
    sel = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    fields = []
    for var in variables:
        da = sel[var["name"]]
        if var["level"] is not None:
            da = da.sel(level=var["level"])
        for attempt in range(retries):
            try:
                arr = da.transpose("time", "latitude", "longitude").values
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                _time.sleep(2 ** (attempt + 1))
        fields.append(arr.astype(np.float32))
    return np.stack(fields, axis=1)  # (time, channel, lat, lon)


def fetch_statics(ds: xr.Dataset, cfg: DataConfig) -> None:
    out = statics_path(cfg)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        name: ds[name].transpose("latitude", "longitude").values.astype(np.float32)
        for name in STATIC_VARIABLES
    }
    data["latitude"] = ds.latitude.values.astype(np.float32)
    data["longitude"] = ds.longitude.values.astype(np.float32)
    np.savez(out, **data)


def build_cache(cfg: DataConfig, years: list[int], verbose: bool = True) -> None:
    ds = open_wb2(cfg.zarr_url)
    fetch_statics(ds, cfg)
    for year in years:
        out = year_path(cfg, year)
        if out.exists():
            if verbose:
                print(f"{year}: cached")
            continue
        t0 = _time.time()
        arr = _fetch_year(ds, year, active_variables(cfg.variable_set))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.rename(out)
        if verbose:
            print(f"{year}: {arr.shape} in {_time.time() - t0:.0f}s")


def load_years(cfg: DataConfig, years: list[int]) -> np.ndarray:
    """Concatenate cached years along time. Years must be contiguous ints."""
    assert years == list(range(years[0], years[-1] + 1)), "years must be contiguous"
    parts = [np.load(year_path(cfg, y), mmap_mode="r") for y in years]
    return np.concatenate(parts, axis=0)


def load_statics(cfg: DataConfig) -> dict[str, np.ndarray]:
    with np.load(statics_path(cfg)) as z:
        return {k: z[k] for k in z.files}

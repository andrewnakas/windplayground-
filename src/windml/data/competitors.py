"""Fetch published WB2 competitor forecasts (64x32) and serve them for scoring.

Each competitor is cached as artifacts/data/competitors/<name>_<year>.npz with:
- forecasts: (n_init, K, C, H, W) float32, NaN where a lead is unavailable
- init_indices: index of each init into the year's 6-hourly time axis

GraphCast 2018 (its training-era forecasts) doubles as training data for the
Phase 9 learned corrector.
"""
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import xarray as xr

from windml.config import VARIABLES, DataConfig
from windml.data.dataset import year_range_times
from windml.eval.forecasters import Forecaster

COMPETITOR_URLS = {
    "graphcast_2020": "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr",
    "graphcast_2018": "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-64x32_equiangular_conservative.zarr",
    "pangu_2020": "gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
    "pangu_2018": "gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
    "hres_2020": "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "hres_2018": "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "gencast_mean_2020": "gs://weatherbench2/datasets/gencast/2020-64x32_equiangular_conservative_mean.zarr",
}


def competitor_path(cfg: DataConfig, name: str, year: int) -> Path:
    return Path(cfg.cache_dir) / "competitors" / f"{name}_{year}.npz"


def fetch_competitor(
    cfg: DataConfig, name: str, year: int, K: int = 20, verbose: bool = True
) -> Path:
    out = competitor_path(cfg, name, year)
    if out.exists():
        return out
    ds = xr.open_zarr(COMPETITOR_URLS[name], storage_options={"token": "anon"})
    year_times = year_range_times((year, year))

    # forecast init times within the year
    ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    init_times = ds.time.values
    # map to indices into the year's 6-hourly axis
    init_indices = ((init_times - year_times[0]) / np.timedelta64(6, "h")).astype(int)

    # available leads within 6h..K*6h
    tds = ds.prediction_timedelta.values
    if np.issubdtype(tds.dtype, np.timedelta64):
        lead_hours = (tds / np.timedelta64(1, "h")).astype(int)
    else:  # stored as plain ints with units attr (hours in WB2 forecast zarrs)
        lead_hours = tds.astype(int)
    keep = (lead_hours >= 6) & (lead_hours <= K * 6)
    ds = ds.isel(prediction_timedelta=np.where(keep)[0])
    lead_slots = (lead_hours[keep] // 6) - 1  # 0-based slot for lead k*6h

    n_init = len(init_times)
    H, W = ds.sizes["latitude"], ds.sizes["longitude"]
    fc = np.full((n_init, K, len(VARIABLES), H, W), np.nan, dtype=np.float32)

    for c, var in enumerate(VARIABLES):
        da = ds[var["name"]]
        if var["level"] is not None:
            da = da.sel(level=var["level"])
        for attempt in range(4):
            try:
                arr = da.transpose(
                    "time", "prediction_timedelta", "latitude", "longitude"
                ).values
                break
            except Exception:
                if attempt == 3:
                    raise
                _time.sleep(2 ** (attempt + 1))
        fc[:, lead_slots, c] = arr
        if verbose:
            print(f"{name} {year}: {var['short']} done")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, forecasts=fc, init_indices=init_indices)
    return out


class CompetitorForecaster(Forecaster):
    def __init__(self, cfg: DataConfig, name: str, year: int, display_name: str | None = None):
        with np.load(competitor_path(cfg, name, year)) as z:
            self.forecasts = z["forecasts"]
            index = z["init_indices"]
        self.lookup = {int(t): i for i, t in enumerate(index)}
        self.name = display_name or name

    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        row = self.lookup.get(init_idx)
        if row is None:
            C, H, W = self.forecasts.shape[2:]
            return np.full((K, C, H, W), np.nan, dtype=np.float32)
        return self.forecasts[row, :K]

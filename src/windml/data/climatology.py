"""Smoothed (day-of-year, hour) climatology from the training years.

WeatherBench uses a climatology per day-of-year and time-of-day, smoothed over a
window of neighboring days, computed on training data only. Used for the
climatology baseline and for ACC anomalies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def doy_hour_index(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """times: datetime64 array -> (day-of-year 0..365, hour-slot 0..3)."""
    days = times.astype("datetime64[D]")
    years = times.astype("datetime64[Y]")
    doy = (days - years.astype("datetime64[D]")).astype(int)
    hours = (times.astype("datetime64[h]") - days.astype("datetime64[h]")).astype(int)
    return doy, hours // 6


def compute_climatology(
    array: np.ndarray, times: np.ndarray, window_days: int = 7
) -> np.ndarray:
    """array: (time, C, H, W); times: datetime64 (same length).

    Returns (366, 4, C, H, W): smoothed mean per (day-of-year, 6h-slot).
    """
    n_t, C, H, W = array.shape
    doy, slot = doy_hour_index(times)
    sums = np.zeros((366, 4, C, H, W), dtype=np.float64)
    counts = np.zeros((366, 4), dtype=np.int64)
    for i in range(n_t):
        sums[doy[i], slot[i]] += array[i]
        counts[doy[i], slot[i]] += 1
    # circular smoothing over day-of-year (+-window_days), weighted by sample
    # counts so days absent from the training span (e.g. Feb 29 outside leap
    # years) contribute nothing instead of diluting the mean with zeros
    idx = np.arange(366)
    sum_w = np.zeros_like(sums)
    count_w = np.zeros_like(counts)
    for offset in range(-window_days, window_days + 1):
        sum_w += sums[(idx + offset) % 366]
        count_w += counts[(idx + offset) % 366]
    count_w = np.maximum(count_w, 1)
    return (sum_w / count_w[:, :, None, None, None]).astype(np.float32)


def climatology_at(clim: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Look up climatology fields for given times -> (len(times), C, H, W)."""
    doy, slot = doy_hour_index(times)
    return clim[doy, slot]


def save_climatology(clim: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, clim)


def load_climatology(path: str | Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")

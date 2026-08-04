"""WeatherBench-style metrics: latitude-weighted RMSE, ACC, and CRPS.

All fields are (..., H, W) with H the latitude axis; weights come from
windml.utils.grid.latitude_weights (mean 1 over the grid).
"""
from __future__ import annotations

import numpy as np


def weighted_mean(x: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Mean over the trailing (H, W) axes, weighted by latitude."""
    return (x * lat_w[:, None]).mean(axis=(-2, -1))


def rmse(pred: np.ndarray, truth: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Latitude-weighted RMSE over (H, W), then averaged over leading axes.

    WeatherBench convention: RMSE is computed per forecast field, then the
    *RMSE values* (not MSEs) are averaged over init times.
    """
    per_field = np.sqrt(weighted_mean((pred - truth) ** 2, lat_w))
    return per_field.mean(axis=0) if per_field.ndim > 0 else per_field


def acc(
    pred: np.ndarray, truth: np.ndarray, clim: np.ndarray, lat_w: np.ndarray
) -> np.ndarray:
    """Anomaly correlation coefficient per field, averaged over the first axis.

    pred/truth/clim: (N, H, W) or (H, W).
    """
    ap = pred - clim
    at = truth - clim
    num = weighted_mean(ap * at, lat_w)
    den = np.sqrt(weighted_mean(ap**2, lat_w) * weighted_mean(at**2, lat_w))
    val = num / np.maximum(den, 1e-12)
    return val.mean(axis=0) if val.ndim > 0 else val


def crps_ensemble(ens: np.ndarray, truth: np.ndarray, lat_w: np.ndarray) -> float:
    """Fair (unbiased) ensemble CRPS, latitude-weighted, averaged over fields.

    ens: (M, N, H, W) ensemble of M members over N fields; truth: (N, H, W).
    CRPS = E|X - y| - 1/(2 M (M-1)) * sum_{i,j} |X_i - X_j|  (fair estimator)
    """
    M = ens.shape[0]
    skill = np.abs(ens - truth[None]).mean(axis=0)
    spread = np.zeros_like(truth, dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            spread += np.abs(ens[i] - ens[j])
    spread *= 2.0 / (M * (M - 1)) if M > 1 else 0.0
    pointwise = skill - 0.5 * spread
    return float(weighted_mean(pointwise, lat_w).mean())


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u**2 + v**2)

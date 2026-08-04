"""Streaming evaluation of a Forecaster over a test period.

Metrics are accumulated per (lead, variable) without storing all forecasts:
RMSE values and per-init ACCs are averaged over init times, following the
WeatherBench convention. Wind speed is scored as a derived variable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windml.config import CHANNELS
from windml.data.climatology import climatology_at
from windml.eval import metrics
from windml.eval.forecasters import Forecaster

U10, V10 = CHANNELS.index("u10"), CHANNELS.index("v10")
SCORED_VARS = CHANNELS + ["wind_speed"]


def evaluate_forecaster(
    fc: Forecaster,
    truth: np.ndarray,
    times: np.ndarray,
    clim: np.ndarray,
    lat_w: np.ndarray,
    K: int = 20,
    init_stride: int = 2,
    init_start: int = 2,
) -> pd.DataFrame:
    """Score `fc` against `truth` (T, C, H, W) at leads 1..K (x6h).

    init_start=2 keeps init times at 00/12 UTC (matching the WB2 competitor
    forecasts and WB2 practice) while leaving t-1 room for two-frame models.
    All forecasters are scored on identical init times.
    """
    n_time = truth.shape[0]
    inits = range(init_start, n_time - K, init_stride)
    n_vars = len(SCORED_VARS)

    rmse_sum = np.zeros((K, n_vars))
    acc_sum = np.zeros((K, n_vars))
    count = np.zeros(K, dtype=np.int64)

    for init_idx in inits:
        pred = fc.forecast(init_idx, K)  # (K, C, H, W); NaN leads are skipped
        lead_ok = np.isfinite(pred).all(axis=(1, 2, 3))  # (K,)
        if not lead_ok.any():
            continue
        target = truth[init_idx + 1 : init_idx + K + 1]
        valid_times = times[init_idx + 1 : init_idx + K + 1]
        clim_fields = climatology_at(clim, valid_times)  # (K, C, H, W)

        pred_ws = metrics.wind_speed(pred[:, U10], pred[:, V10])
        target_ws = metrics.wind_speed(target[:, U10], target[:, V10])
        clim_ws = metrics.wind_speed(clim_fields[:, U10], clim_fields[:, V10])

        pred_all = np.concatenate([pred, pred_ws[:, None]], axis=1)
        target_all = np.concatenate([target, target_ws[:, None]], axis=1)
        clim_all = np.concatenate([clim_fields, clim_ws[:, None]], axis=1)

        err = metrics.weighted_mean((pred_all - target_all) ** 2, lat_w)  # (K, V)
        ap = pred_all - clim_all
        at = target_all - clim_all
        num = metrics.weighted_mean(ap * at, lat_w)
        den = np.sqrt(
            metrics.weighted_mean(ap**2, lat_w) * metrics.weighted_mean(at**2, lat_w)
        )
        acc_field = num / np.maximum(den, 1e-12)
        rmse_sum[lead_ok] += np.sqrt(err[lead_ok])
        acc_sum[lead_ok] += acc_field[lead_ok]
        count += lead_ok.astype(np.int64)

    rows = []
    for k in range(K):
        for v, var in enumerate(SCORED_VARS):
            rows.append(
                {
                    "model": fc.name,
                    "variable": var,
                    "lead_h": (k + 1) * 6,
                    "rmse": rmse_sum[k, v] / max(count[k], 1),
                    "acc": acc_sum[k, v] / max(count[k], 1),
                    "n_inits": int(count[k]),
                }
            )
    return pd.DataFrame(rows)

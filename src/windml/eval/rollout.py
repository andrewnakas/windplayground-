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
    batch_size: int = 32,
) -> pd.DataFrame:
    """Score `fc` against `truth` (T, C, H, W) at leads 1..K (x6h).

    init_start=2 keeps init times at 00/12 UTC (matching the WB2 competitor
    forecasts and WB2 practice) while leaving t-1 room for two-frame models.
    All forecasters are scored on identical init times.
    """
    n_time = truth.shape[0]
    inits = list(range(init_start, n_time - K, init_stride))
    n_vars = len(SCORED_VARS)

    rmse_sum = np.zeros((K, n_vars))
    acc_sum = np.zeros((K, n_vars))
    count = np.zeros((K, n_vars), dtype=np.int64)

    for start in range(0, len(inits), batch_size):
        chunk = inits[start : start + batch_size]
        preds = fc.forecast_batch(chunk, K)  # (B, K, C, H, W); NaN leads skipped
        for pred, init_idx in zip(preds, chunk):
            # A model may predict only some variables (RT2021 forecasts
            # z500/t850/t2m and nothing else) and only some leads (a direct
            # 72h model). So validity is per (lead, variable), not per lead:
            # masking whole leads would throw away the three variables an
            # RT2021 model *does* forecast and silently report all-NaN.
            if not np.isfinite(pred).any():
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

            # (K, V) validity, computed after the wind-speed column is
            # appended so a NaN u10/v10 correctly invalidates wind_speed too.
            ok = np.isfinite(pred_all).all(axis=(2, 3))

            err = metrics.weighted_mean((pred_all - target_all) ** 2, lat_w)  # (K, V)
            ap = pred_all - clim_all
            at = target_all - clim_all
            num = metrics.weighted_mean(ap * at, lat_w)
            den = np.sqrt(
                metrics.weighted_mean(ap**2, lat_w) * metrics.weighted_mean(at**2, lat_w)
            )
            acc_field = num / np.maximum(den, 1e-12)
            rmse_sum += np.where(ok, np.sqrt(np.where(ok, err, 0.0)), 0.0)
            acc_sum += np.where(ok, np.where(ok, acc_field, 0.0), 0.0)
            count += ok.astype(np.int64)

    rows = []
    for k in range(K):
        # a lead with no valid init times (e.g. odd 6h leads for a 12-hourly
        # model such as GenCast) scores NaN, not 0 -- 0 would read as perfect
        for v, var in enumerate(SCORED_VARS):
            n = count[k, v]
            rows.append(
                {
                    "model": fc.name,
                    "variable": var,
                    "lead_h": (k + 1) * 6,
                    "rmse": rmse_sum[k, v] / n if n else np.nan,
                    "acc": acc_sum[k, v] / n if n else np.nan,
                    "n_inits": int(n),
                }
            )
    return pd.DataFrame(rows)

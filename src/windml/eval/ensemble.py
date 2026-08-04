"""Ensemble construction and probabilistic scoring (GenCast-inspired, toy scale).

Two cheap ways to turn a deterministic forecaster into an ensemble:
- **IC perturbation**: perturb the initial state with scaled climatological
  noise, roll out each member (spread grows with lead time, like a real EPS).
- **Gaussian dressing**: add a per-(variable, lead) spread calibrated on a
  training year, i.e. the classic statistical post-processing ensemble.

Both are scored with the fair CRPS in windml.eval.metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windml.config import CHANNELS
from windml.data.climatology import climatology_at
from windml.eval import metrics
from windml.eval.forecasters import Forecaster

U10, V10 = CHANNELS.index("u10"), CHANNELS.index("v10")


class DressedEnsemble:
    """Adds calibrated Gaussian noise to a deterministic forecast.

    spread: (K, C) per-lead, per-variable standard deviations, typically the
    forecaster's own RMSE on a *training* year (the perfectly-calibrated
    dressing under a Gaussian error model).
    """

    def __init__(self, base: Forecaster, spread: np.ndarray, n_members: int = 8, seed: int = 0):
        self.base = base
        self.spread = spread
        self.M = n_members
        self.rng = np.random.default_rng(seed)
        self.name = f"{base.name}+dressed{n_members}"

    def ensemble(self, init_idx: int, K: int) -> np.ndarray:
        """(M, K, C, H, W) ensemble members."""
        mean = self.base.forecast(init_idx, K)
        noise = self.rng.normal(size=(self.M, *mean.shape)).astype(np.float32)
        return mean[None] + noise * self.spread[None, :K, :, None, None]


def spread_from_scores(scores: pd.DataFrame, K: int) -> np.ndarray:
    """Build a (K, C) spread array from an evaluator results CSV (RMSE rows)."""
    spread = np.zeros((K, len(CHANNELS)), dtype=np.float32)
    for c, var in enumerate(CHANNELS):
        sub = scores[scores.variable == var].sort_values("lead_h")
        vals = sub.rmse.to_numpy()[:K]
        spread[: len(vals), c] = vals
    return spread


def evaluate_ensemble(
    ens_source,
    truth: np.ndarray,
    times: np.ndarray,
    clim: np.ndarray,
    lat_w: np.ndarray,
    K: int = 20,
    init_stride: int = 2,
    init_start: int = 2,
    name: str | None = None,
) -> pd.DataFrame:
    """CRPS + ensemble-mean RMSE + spread-skill ratio per (lead, variable)."""
    n_time = truth.shape[0]
    inits = range(init_start, n_time - K, init_stride)
    scored = CHANNELS + ["wind_speed"]
    crps_sum = np.zeros((K, len(scored)))
    rmse_sum = np.zeros((K, len(scored)))
    spread_sum = np.zeros((K, len(scored)))
    count = np.zeros(K, dtype=np.int64)

    for init_idx in inits:
        ens = ens_source.ensemble(init_idx, K)  # (M, K, C, H, W)
        ok = np.isfinite(ens).all(axis=(0, 2, 3, 4))
        if not ok.any():
            continue
        target = truth[init_idx + 1 : init_idx + K + 1]
        ws_ens = metrics.wind_speed(ens[:, :, U10], ens[:, :, V10])[:, :, None]
        ens_all = np.concatenate([ens, ws_ens], axis=2)
        ws_t = metrics.wind_speed(target[:, U10], target[:, V10])[:, None]
        tgt_all = np.concatenate([target, ws_t], axis=1)

        for k in range(K):
            if not ok[k]:
                continue
            for v in range(len(scored)):
                members = ens_all[:, k, v]
                crps_sum[k, v] += metrics.crps_ensemble(
                    members, tgt_all[k, v], lat_w
                )
                mean = members.mean(axis=0)
                rmse_sum[k, v] += np.sqrt(
                    metrics.weighted_mean((mean - tgt_all[k, v]) ** 2, lat_w)
                )
                spread_sum[k, v] += np.sqrt(
                    metrics.weighted_mean(members.var(axis=0, ddof=1), lat_w)
                )
        count += ok.astype(np.int64)

    rows = []
    for k in range(K):
        n = max(count[k], 1)
        for v, var in enumerate(scored):
            rmse = rmse_sum[k, v] / n
            spread = spread_sum[k, v] / n
            rows.append(
                {
                    "model": name or getattr(ens_source, "name", "ensemble"),
                    "variable": var,
                    "lead_h": (k + 1) * 6,
                    "crps": crps_sum[k, v] / n,
                    "rmse": rmse,
                    "spread": spread,
                    "spread_skill": spread / max(rmse, 1e-9),
                    "n_inits": int(count[k]),
                }
            )
    return pd.DataFrame(rows)

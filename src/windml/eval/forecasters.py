"""Forecaster interface + reference baselines.

A Forecaster produces physical-unit forecasts for the K leads following an
init index into a test-period array. The rollout evaluator only sees this
interface, so trained models, trivial baselines, and stored competitor
forecasts are all scored identically.
"""
from __future__ import annotations

import numpy as np
import torch

from windml.data.climatology import climatology_at
from windml.data.normalization import Normalizer


class Forecaster:
    name: str = "forecaster"

    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        """Return (K, C, H, W) physical-unit forecasts for leads 1..K."""
        raise NotImplementedError

    def forecast_batch(self, init_indices: list[int], K: int) -> np.ndarray:
        """Return (B, K, C, H, W) for several init times.

        The default loops over `forecast`; ModelForecaster overrides this to
        roll out all inits together, which is far faster on CPU because each
        step becomes one batched matmul instead of B tiny ones.
        """
        return np.stack([self.forecast(i, K) for i in init_indices])


class PersistenceForecaster(Forecaster):
    name = "persistence"

    def __init__(self, array: np.ndarray):
        self.array = array

    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        state = self.array[init_idx]
        return np.repeat(state[None], K, axis=0)


class ClimatologyForecaster(Forecaster):
    name = "climatology"

    def __init__(self, clim: np.ndarray, times: np.ndarray):
        self.clim = clim
        self.times = times

    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        valid_times = self.times[init_idx + 1 : init_idx + K + 1]
        return climatology_at(self.clim, valid_times)


class ModelForecaster(Forecaster):
    """Autoregressive rollout of a residual-predicting torch model."""

    def __init__(self, model: torch.nn.Module, dataset, normalizer: Normalizer, name: str):
        self.model = model.eval()
        self.ds = dataset  # Era5Dataset over the test span (provides input_at)
        self.norm = normalizer
        self.name = name

    @torch.no_grad()
    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        return self.forecast_batch([init_idx], K)[0]

    @torch.no_grad()
    def forecast_batch(self, init_indices: list[int], K: int) -> np.ndarray:
        """Roll all init times forward together: one batched forward per lead."""
        idx = np.asarray(init_indices)
        C, H, W = self.ds.array.shape[1:]
        B = len(idx)
        curr = self.ds.array[idx].astype(np.float32)  # (B, C, H, W)
        prev = self.ds.array[idx - 1].astype(np.float32) if self.ds.two_frame else None
        static = np.broadcast_to(self.ds.static[None], (B, *self.ds.static.shape))
        out = np.empty((B, K, C, H, W), dtype=np.float32)

        for k in range(K):
            t_abs = idx + k  # timestamps of the *current* states
            frames = [self.norm.norm_state(curr)]
            if prev is not None:
                frames.append(self.norm.norm_state(prev))
            tf = np.broadcast_to(
                self.ds.time_feats[t_abs][:, :, None, None], (B, 4, H, W)
            )
            x = np.concatenate(frames + [static, tf], axis=1).astype(np.float32)
            resid = self.norm.denorm_residual(self.model(torch.from_numpy(x)).numpy())
            prev_state = curr
            curr = curr + resid
            if prev is not None:
                prev = prev_state
            out[:, k] = curr
        return out


class StoredForecaster(Forecaster):
    """Pre-computed forecasts indexed as (init_time, lead) -> (C, H, W).

    Used for the published WB2 competitor forecasts (GraphCast, Pangu, HRES...)
    regridded to 64x32. `forecasts` maps our init indices to stored arrays;
    missing (init, lead) entries are NaN and skipped by the evaluator.
    """

    def __init__(self, forecasts: np.ndarray, name: str):
        self.forecasts = forecasts  # (n_init, K, C, H, W)
        self.name = name

    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        return self.forecasts[init_idx, :K]

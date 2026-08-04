"""Learned post-processing of a frozen competitor forecast (Phase 9).

The corrector sees a stored forecast field at lead k plus statics, valid-time
features, and the lead time, and predicts the (state-normalized) error
truth - forecast. Zero-initialized head => starts as the identity correction.
Trained on the competitor's 2018 forecasts, evaluated on 2020: fully
out-of-sample, same protocol as every other forecaster.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from windml.config import DataConfig
from windml.data.build_cache import load_years
from windml.data.competitors import CompetitorForecaster, competitor_path
from windml.data.dataset import build_static_channels, year_range_times
from windml.data.normalization import Normalizer
from windml.eval.forecasters import Forecaster
from windml.utils.grid import time_encodings


class CorrectorDataset(Dataset):
    """(init, lead) pairs from a stored competitor forecast vs ERA5 truth."""

    def __init__(self, cfg: DataConfig, competitor: str, year: int, norm: Normalizer):
        with np.load(competitor_path(cfg, competitor, year)) as z:
            self.forecasts = z["forecasts"]  # (n_init, K, C, H, W)
            self.init_indices = z["init_indices"]
        self.truth = load_years(cfg, [year])
        self.times = year_range_times((year, year))
        self.norm = norm
        self.static = build_static_channels(cfg)
        hours = (self.times - np.datetime64("1970-01-01T00", "h")) / np.timedelta64(1, "h")
        self.time_feats = time_encodings(hours.astype(np.float64))
        self.K = self.forecasts.shape[1]
        # valid (row, lead) pairs: finite forecast, target inside the year
        self.pairs = [
            (r, k)
            for r, t0 in enumerate(self.init_indices)
            for k in range(self.K)
            if t0 + k + 1 < len(self.times)
            and np.isfinite(self.forecasts[r, k, 0, 0, 0])
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def n_input_channels(self) -> int:
        C = self.forecasts.shape[2]
        return C + self.static.shape[0] + 4 + 1  # + lead channel

    def build_input(self, fc_field: np.ndarray, valid_t: int, lead_k: int) -> np.ndarray:
        H, W = fc_field.shape[-2:]
        fc_norm = self.norm.norm_state(fc_field[None])[0]
        tf = np.broadcast_to(self.time_feats[valid_t][:, None, None], (4, H, W))
        lead = np.full((1, H, W), (lead_k + 1) / self.K, dtype=np.float32)
        return np.concatenate([fc_norm, self.static, tf, lead]).astype(np.float32)

    def __getitem__(self, idx: int):
        r, k = self.pairs[idx]
        valid_t = int(self.init_indices[r]) + k + 1
        fc_field = self.forecasts[r, k]
        x = self.build_input(fc_field, valid_t, k)
        err = (self.truth[valid_t] - fc_field)[None]
        y = (err / self.norm.std).astype(np.float32)[0]
        return torch.from_numpy(x), torch.from_numpy(y)


class CorrectedForecaster(Forecaster):
    """Applies a trained corrector on top of a stored competitor forecast."""

    def __init__(
        self,
        base: CompetitorForecaster,
        model: torch.nn.Module,
        ds: CorrectorDataset,
        name: str,
    ):
        self.base = base
        self.model = model.eval()
        self.ds = ds  # provides build_input/statics/time feats for the eval year
        self.name = name

    @torch.no_grad()
    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        raw = self.base.forecast(init_idx, K)
        out = raw.copy()
        for k in range(K):
            if not np.isfinite(raw[k]).all():
                continue
            x = self.ds.build_input(raw[k], init_idx + k + 1, k)
            corr = self.model(torch.from_numpy(x[None]))[0].numpy()
            out[k] = raw[k] + corr * self.ds.norm.std[0]
        return out

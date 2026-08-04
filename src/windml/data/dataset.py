"""Torch dataset over the cached ERA5 arrays.

A sample maps the (normalized) state at time t — optionally with the t-6h state —
plus static/positional/time channels, to K successive residual targets
(x_{t+k} - x_{t+k-1}) / diff_std for k = 1..K.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from windml.config import DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.normalization import Normalizer
from windml.utils.grid import spatial_encodings, time_encodings


def year_range_times(years: tuple[int, int]) -> np.ndarray:
    """6-hourly datetime64 timestamps covering [start_year, end_year]."""
    start = np.datetime64(f"{years[0]}-01-01T00", "h")
    end = np.datetime64(f"{years[1] + 1}-01-01T00", "h")
    return np.arange(start, end, np.timedelta64(6, "h"))


def build_static_channels(cfg: DataConfig) -> np.ndarray:
    """(5, H, W): normalized land-sea mask + orography, and 3 spherical encodings."""
    statics = load_statics(cfg)
    lsm = statics["land_sea_mask"]
    oro = statics["geopotential_at_surface"]
    oro = (oro - oro.mean()) / (oro.std() + 1e-8)
    enc = spatial_encodings(statics["latitude"], statics["longitude"])
    return np.concatenate([np.stack([lsm, oro]), enc]).astype(np.float32)


class Era5Dataset(Dataset):
    """In-RAM dataset over contiguous cached years.

    Input channels: C dynamic (t) [+ C dynamic (t-6h)] + 5 static + 4 time = model input.
    Targets: (K, C, H, W) normalized residuals.
    """

    def __init__(
        self,
        cfg: DataConfig,
        years: tuple[int, int],
        normalizer: Normalizer,
        rollout_steps: int = 1,
        two_frame: bool = True,
    ):
        self.array = np.ascontiguousarray(load_years(cfg, list(range(years[0], years[1] + 1))))
        self.times = year_range_times(years)
        assert len(self.times) == self.array.shape[0], (
            f"time axis mismatch: {len(self.times)} vs {self.array.shape[0]}"
        )
        self.norm = normalizer
        self.K = rollout_steps
        self.two_frame = two_frame
        self.static = build_static_channels(cfg)  # (5, H, W)
        hours = (self.times - np.datetime64("1970-01-01T00", "h")) / np.timedelta64(1, "h")
        self.time_feats = time_encodings(hours.astype(np.float64))  # (T, 4)
        self.margin = 1 if two_frame else 0

    def __len__(self) -> int:
        return self.array.shape[0] - self.K - self.margin

    @property
    def n_input_channels(self) -> int:
        C = self.array.shape[1]
        return C * (2 if self.two_frame else 1) + self.static.shape[0] + 4

    def input_at(self, t: int) -> np.ndarray:
        """Assemble the model input for absolute time index t."""
        C, H, W = self.array.shape[1:]
        frames = [self.norm.norm_state(self.array[t : t + 1])[0]]
        if self.two_frame:
            frames.append(self.norm.norm_state(self.array[t - 1 : t])[0])
        tf = np.broadcast_to(self.time_feats[t][:, None, None], (4, H, W))
        return np.concatenate(frames + [self.static, tf]).astype(np.float32)

    def __getitem__(self, idx: int):
        t = idx + self.margin
        x = self.input_at(t)
        future = self.array[t : t + self.K + 1].astype(np.float32)
        residuals = self.norm.norm_residual(np.diff(future, axis=0))
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(residuals))

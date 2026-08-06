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
    """Constant per-pixel channels.

    Default (5, H, W): normalized land-sea mask + orography, and 3 spherical
    encodings.

    For the `rt2021` sets this returns exactly (3, H, W) -- land-sea mask,
    orography, latitude -- because that is the paper's constant set, and
    because the extra encodings would push the input past the 117 channels it
    reports. Latitude is given as sin(lat), which is the same information on a
    bounded scale.
    """
    statics = load_statics(cfg)
    lsm = statics["land_sea_mask"]
    oro = statics["geopotential_at_surface"]
    oro = (oro - oro.mean()) / (oro.std() + 1e-8)
    if cfg.variable_set.startswith("rt2021"):
        lat = np.sin(np.deg2rad(statics["latitude"]))[:, None]
        lat = np.broadcast_to(lat, lsm.shape)
        return np.stack([lsm, oro, lat]).astype(np.float32)
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
        direct_steps: int | None = None,
        n_frames: int | None = None,
    ):
        """direct_steps: predict the state this many 6h steps ahead in ONE shot
        instead of rolling the 6h model forward. This is how Rasp & Thuerey's
        WeatherBench ResNet reaches 3 days, and it avoids the error the
        autoregressive rollout accumulates over 12 steps.

        n_frames: how many consecutive past states to stack (t, t-6h, t-12h...).
        RT2021 uses 3; our other models use 2. Defaults to the `two_frame`
        setting so existing configs keep their behaviour unchanged.
        """
        self.array = np.ascontiguousarray(load_years(cfg, list(range(years[0], years[1] + 1))))
        self.times = year_range_times(years)
        assert len(self.times) == self.array.shape[0], (
            f"time axis mismatch: {len(self.times)} vs {self.array.shape[0]}"
        )
        self.norm = normalizer
        self.K = rollout_steps
        self.n_frames = n_frames if n_frames is not None else (2 if two_frame else 1)
        self.two_frame = self.n_frames >= 2
        self.direct_steps = direct_steps
        self.static = build_static_channels(cfg)
        # RT2021 has no time-of-year/day features: TISR already carries that
        # signal, and adding them would exceed the paper's 117 input channels.
        self.use_time_feats = not cfg.variable_set.startswith("rt2021")
        hours = (self.times - np.datetime64("1970-01-01T00", "h")) / np.timedelta64(1, "h")
        self.time_feats = time_encodings(hours.astype(np.float64))  # (T, 4)
        # earliest usable index: enough history for every stacked frame
        self.margin = self.n_frames - 1
        # scale for the direct target: spread of L-step differences, estimated
        # on a sample so it is cheap and stable
        self.direct_std = (
            self._estimate_direct_std(direct_steps) if direct_steps else None
        )

    def _estimate_direct_std(self, steps: int, n_samples: int = 1500) -> np.ndarray:
        rng = np.random.default_rng(0)
        hi = self.array.shape[0] - steps - 1
        idx = rng.choice(hi, size=min(n_samples, hi), replace=False)
        diffs = self.array[idx + steps].astype(np.float32) - self.array[idx].astype(np.float32)
        return diffs.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)[
            None, :, None, None
        ]

    @property
    def horizon(self) -> int:
        """Timesteps of future data a sample needs."""
        return self.direct_steps if self.direct_steps else self.K

    def __len__(self) -> int:
        return self.array.shape[0] - self.horizon - self.margin

    @property
    def n_input_channels(self) -> int:
        C = self.array.shape[1]
        n_time = 4 if self.use_time_feats else 0
        return C * self.n_frames + self.static.shape[0] + n_time

    def input_at(self, t: int) -> np.ndarray:
        """Assemble the model input for absolute time index t.

        Frames are stacked most-recent-first (t, t-6h, t-12h, ...), which is
        the order `channel_index_map` in models/grow.py assumes when mapping a
        pretrained stack onto a wider fine-tuning one.
        """
        H, W = self.array.shape[2:]
        frames = [
            self.norm.norm_state(self.array[t - k : t - k + 1])[0]
            for k in range(self.n_frames)
        ]
        parts = frames + [self.static]
        if self.use_time_feats:
            parts.append(np.broadcast_to(self.time_feats[t][:, None, None], (4, H, W)))
        return np.concatenate(parts).astype(np.float32)

    def __getitem__(self, idx: int):
        t = idx + self.margin
        x = self.input_at(t)
        if self.direct_steps:
            # one shot to the target lead: (x_{t+L} - x_t) / std_L
            delta = (self.array[t + self.direct_steps] - self.array[t]).astype(np.float32)
            target = (delta[None] / self.direct_std)[0]
            return torch.from_numpy(x), torch.from_numpy(target[None])
        future = self.array[t : t + self.K + 1].astype(np.float32)
        residuals = self.norm.norm_residual(np.diff(future, axis=0))
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(residuals))

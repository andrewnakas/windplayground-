"""Latitude-weighted MSE in normalized residual space."""
from __future__ import annotations

import numpy as np
import torch

from windml.config import CHANNELS


def make_channel_weights(overrides: dict[str, float] | None = None) -> torch.Tensor:
    w = torch.ones(len(CHANNELS))
    for name, val in (overrides or {}).items():
        w[CHANNELS.index(name)] = val
    return w / w.mean()


class LatWeightedMSE(torch.nn.Module):
    def __init__(self, lat_weights: np.ndarray, channel_weights: torch.Tensor | None = None):
        super().__init__()
        lw = torch.from_numpy(lat_weights.astype(np.float32))[None, None, :, None]
        self.register_buffer("lat_w", lw)
        cw = channel_weights if channel_weights is not None else torch.ones(len(CHANNELS))
        self.register_buffer("chan_w", cw[None, :, None, None])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = (pred - target) ** 2 * self.lat_w * self.chan_w
        return err.mean()

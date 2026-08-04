"""Per-channel normalization statistics computed on the training years.

Two kinds of statistics, both per channel:
- mean/std of the raw fields (input normalization)
- std of the 6h differences (target scaling for residual prediction, per GraphCast)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np




def compute_stats(train_array: np.ndarray, channels: list[str] | None = None) -> dict:
    """train_array: (time, channel, lat, lon) float32."""
    x = train_array
    mean = x.mean(axis=(0, 2, 3), dtype=np.float64)
    std = x.std(axis=(0, 2, 3), dtype=np.float64)
    diff = np.diff(x[:, :, :, :], axis=0)
    diff_std = diff.std(axis=(0, 2, 3), dtype=np.float64)
    if channels is None:
        from windml.config import CHANNELS as channels
    return {
        "channels": list(channels),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "diff_std": diff_std.tolist(),
    }


def save_stats(stats: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path) -> dict:
    stats = json.loads(Path(path).read_text())
    for k in ("mean", "std", "diff_std"):
        stats[k] = np.asarray(stats[k], dtype=np.float32)
    return stats


class Normalizer:
    """Normalize states and residual targets; denormalize predictions."""

    def __init__(self, stats: dict):
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[None, :, None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[None, :, None, None]
        self.diff_std = np.asarray(stats["diff_std"], dtype=np.float32)[None, :, None, None]

    def norm_state(self, x):
        return (x - self.mean) / self.std

    def denorm_state(self, x):
        return x * self.std + self.mean

    def norm_residual(self, dx):
        return dx / self.diff_std

    def denorm_residual(self, dx):
        return dx * self.diff_std

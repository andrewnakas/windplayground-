"""Per-channel normalization statistics computed on the training years.

Two kinds of statistics, both per channel:
- mean/std of the raw fields (input normalization)
- std of the 6h differences (target scaling for residual prediction, per GraphCast)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np




def compute_stats_streaming(
    blocks, channels: list[str] | None = None, chunk: int = 1000
) -> dict:
    """Same statistics as compute_stats, over an iterable of time blocks.

    Lets the caller feed one year at a time (~240 MB) instead of concatenating
    39 years (~8.7 GB) into RAM first. Differences are carried across block
    boundaries, so the result matches processing the whole series at once.
    """
    acc = _StatAccumulator()
    for block in blocks:
        acc.update(np.asarray(block), chunk)
    return acc.finalize(channels)


class _StatAccumulator:
    def __init__(self):
        self.total = self.total_sq = self.d_total = self.d_total_sq = None
        self.n = self.n_diff = 0
        self.tail = None

    def update(self, block: np.ndarray, chunk: int = 1000) -> None:
        C = block.shape[1]
        if self.total is None:
            z = lambda: np.zeros(C, dtype=np.float64)  # noqa: E731
            self.total, self.total_sq = z(), z()
            self.d_total, self.d_total_sq = z(), z()
        for start in range(0, block.shape[0], chunk):
            x = np.asarray(block[start : start + chunk], dtype=np.float64)
            self.total += x.sum(axis=(0, 2, 3))
            self.total_sq += np.square(x).sum(axis=(0, 2, 3))
            self.n += x.shape[0] * x.shape[2] * x.shape[3]
            d = np.diff(x, axis=0) if self.tail is None else np.diff(
                np.concatenate([self.tail[None], x]), axis=0
            )
            self.d_total += d.sum(axis=(0, 2, 3))
            self.d_total_sq += np.square(d).sum(axis=(0, 2, 3))
            self.n_diff += d.shape[0] * d.shape[2] * d.shape[3]
            self.tail = x[-1]

    def finalize(self, channels: list[str] | None) -> dict:
        mean = self.total / self.n
        var = np.maximum(self.total_sq / self.n - mean**2, 0.0)
        d_mean = self.d_total / self.n_diff
        d_var = np.maximum(self.d_total_sq / self.n_diff - d_mean**2, 0.0)
        if channels is None:
            from windml.config import CHANNELS as channels
        return {
            "channels": list(channels),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "diff_std": np.sqrt(d_var).tolist(),
        }


def compute_stats(
    train_array: np.ndarray, channels: list[str] | None = None, chunk: int = 1000
) -> dict:
    """train_array: (time, channel, lat, lon) float32.

    Accumulates in time chunks rather than reducing the whole array at once:
    numpy allocates a float64 temporary the size of the input for a reduction
    with dtype=float64, which is 17 GB for 39 years of the 20-channel set.
    """
    n_t, C = train_array.shape[:2]
    total = np.zeros(C, dtype=np.float64)
    total_sq = np.zeros(C, dtype=np.float64)
    d_total = np.zeros(C, dtype=np.float64)
    d_total_sq = np.zeros(C, dtype=np.float64)
    n = 0
    n_diff = 0
    tail = None  # last frame of the previous chunk, so diffs span boundaries

    for start in range(0, n_t, chunk):
        x = np.asarray(train_array[start : start + chunk], dtype=np.float64)
        total += x.sum(axis=(0, 2, 3))
        total_sq += np.square(x).sum(axis=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]

        d = np.diff(x, axis=0) if tail is None else np.diff(
            np.concatenate([tail[None], x]), axis=0
        )
        d_total += d.sum(axis=(0, 2, 3))
        d_total_sq += np.square(d).sum(axis=(0, 2, 3))
        n_diff += d.shape[0] * d.shape[2] * d.shape[3]
        tail = x[-1]

    mean = total / n
    var = np.maximum(total_sq / n - mean**2, 0.0)
    d_mean = d_total / n_diff
    d_var = np.maximum(d_total_sq / n_diff - d_mean**2, 0.0)

    if channels is None:
        from windml.config import CHANNELS as channels
    return {
        "channels": list(channels),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "diff_std": np.sqrt(d_var).tolist(),
    }


def save_stats(stats: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path) -> dict:
    stats = json.loads(Path(path).read_text())
    for k in ("mean", "std", "diff_std"):
        stats[k] = np.asarray(stats[k], dtype=np.float32)
    return stats


def log_transform(x: np.ndarray, channels: list[str]) -> np.ndarray:
    """Rasp & Thuerey's precipitation transform: log(eps+PR) - log(eps).

    Applied to the channels named in `LOG_TRANSFORM_VARIABLES`, before any
    statistics are computed. Subtracting log(eps) is what keeps exact zeros at
    exact zero, so "no rain" stays representable after the transform; combined
    with standardizing by std alone (see `Normalizer`), the variable's lower
    bound survives normalization.

    The paper calls this crucial: without it the distribution is skewed enough
    that the network simply learns to predict zeros everywhere.

    Returns a copy; `x` is (time, channel, lat, lon).
    """
    from windml.config import LOG_TRANSFORM_VARIABLES, PRECIP_LOG_EPSILON

    idx = [i for i, c in enumerate(channels) if c in LOG_TRANSFORM_VARIABLES]
    if not idx:
        return x
    out = np.array(x, dtype=np.float32, copy=True)
    eps = PRECIP_LOG_EPSILON
    # log(eps+PR) - log(eps) == log1p(PR/eps), but evaluated as the difference
    # of two logs in float32 it leaves ~1e-7 of residue at PR=0, which defeats
    # the whole purpose. log1p(0) is exactly 0, and it is more accurate for the
    # small values that dominate precipitation.
    #
    # Accumulated precipitation is non-negative in principle; tiny negatives do
    # show up from the regridding, and they would produce NaN here.
    out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0) / eps)
    return out


def inverse_log_transform(x: np.ndarray, channels: list[str]) -> np.ndarray:
    """Undo `log_transform`, for reporting precipitation in physical units."""
    from windml.config import LOG_TRANSFORM_VARIABLES, PRECIP_LOG_EPSILON

    idx = [i for i, c in enumerate(channels) if c in LOG_TRANSFORM_VARIABLES]
    if not idx:
        return x
    out = np.array(x, dtype=np.float32, copy=True)
    eps = PRECIP_LOG_EPSILON
    # Inverse of log1p(PR/eps); expm1(0) is exactly 0, so zeros survive both ways.
    out[:, idx] = eps * np.expm1(out[:, idx])
    return out


class Normalizer:
    """Normalize states and residual targets; denormalize predictions.

    Channels listed in `LOG_TRANSFORM_VARIABLES` are scaled by their standard
    deviation but have **no mean subtracted**, which is what preserves the
    zero lower bound of log-transformed precipitation. Which channels those are
    is read from `stats["channels"]`, so callers need not pass anything extra --
    and stats files that predate the RT2021 variable set are unaffected, since
    they contain no such channel.
    """

    def __init__(self, stats: dict):
        from windml.config import LOG_TRANSFORM_VARIABLES

        mean = np.asarray(stats["mean"], dtype=np.float32).copy()
        for i, channel in enumerate(stats.get("channels", [])):
            if channel in LOG_TRANSFORM_VARIABLES:
                mean[i] = 0.0
        self.mean = mean[None, :, None, None]
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

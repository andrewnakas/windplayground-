"""Linear ridge baseline: a learned skill floor above persistence.

Fits a shared-across-pixels ridge map from the normalized current state
(8 channels + 5 static + 4 time features per pixel) to the normalized 6h
residual, by solving the normal equations on a subsample of training pairs.
Applied autoregressively like any other model.
"""
from __future__ import annotations

import numpy as np
import torch

from windml.data.dataset import Era5Dataset


class LinearModel(torch.nn.Module):
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.register_buffer("W", torch.from_numpy(weights.astype(np.float32)))
        self.register_buffer("b", torch.from_numpy(bias.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, H, W) -> per-pixel linear map -> (B, C, H, W)
        return torch.einsum("bfhw,cf->bchw", x, self.W) + self.b[None, :, None, None]


def fit_linear(train_ds: Era5Dataset, n_samples: int = 2000, ridge: float = 1e-3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_ds), size=min(n_samples, len(train_ds)), replace=False)
    xs, ys = [], []
    for i in idx:
        x, y = train_ds[int(i)]
        xs.append(x.numpy().reshape(x.shape[0], -1).T)  # (HW, F)
        ys.append(y.numpy()[0].reshape(y.shape[1], -1).T)  # (HW, C)
    X = np.concatenate(xs)  # (N*HW, F)
    Y = np.concatenate(ys)  # (N*HW, C)
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    XtX = X.T @ X + ridge * X.shape[0] * np.eye(X.shape[1], dtype=np.float64)
    XtY = X.T @ Y
    Wb = np.linalg.solve(XtX, XtY)  # (F+1, C)
    return LinearModel(Wb[:-1].T, Wb[-1])

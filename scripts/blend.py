"""Per-(variable, lead) affine blend of competitor forecasts (superensemble).

Weights are fit by latitude-weighted least squares on 2018 forecasts
(GraphCast + Pangu + HRES vs ERA5), then applied to the same members' 2020
forecasts and scored with the standard evaluator.

Usage: python scripts/blend.py [--members graphcast pangu hres]
"""
from __future__ import annotations

import argparse

import numpy as np

from windml.config import ARTIFACTS, DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.climatology import load_climatology
from windml.data.competitors import CompetitorForecaster
from windml.data.dataset import year_range_times
from windml.eval.forecasters import Forecaster
from windml.eval.rollout import evaluate_forecaster
from windml.utils.grid import latitude_weights

CLIM_PATH = ARTIFACTS / "climatology" / "clim_train.npy"


class BlendForecaster(Forecaster):
    def __init__(self, members: list[CompetitorForecaster], weights: np.ndarray, name: str):
        self.members = members
        self.weights = weights  # (K, C, n_members + 1) affine: w @ members + b
        self.name = name

    def forecast(self, init_idx: int, K: int) -> np.ndarray:
        preds = [m.forecast(init_idx, K) for m in self.members]
        out = np.full_like(preds[0], np.nan)
        for k in range(K):
            fields = [p[k] for p in preds]
            if not all(np.isfinite(f).all() for f in fields):
                continue
            w = self.weights[k]  # (C, M+1)
            stack = np.stack(fields)  # (M, C, H, W)
            out[k] = np.einsum("mchw,cm->chw", stack, w[:, :-1]) + w[:, -1:, None]
        return out


def fit_weights(
    cfg: DataConfig, member_names: list[str], year: int, K: int, lat_w: np.ndarray
) -> np.ndarray:
    members = [CompetitorForecaster(cfg, f"{m}_{year}", year) for m in member_names]
    truth = np.asarray(load_years(cfg, [year]), dtype=np.float32)
    n_time = truth.shape[0]
    common = sorted(set.intersection(*(set(m.lookup) for m in members)))
    common = [t for t in common if 2 <= t < n_time - K]
    M = len(members)
    C = truth.shape[1]
    w_out = np.zeros((K, C, M + 1), dtype=np.float32)
    sqrt_w = np.sqrt(lat_w)[:, None]

    for k in range(K):
        # gather all members: (M, N, C, H, W)
        stack = np.stack(
            [np.stack([m.forecast(t, K)[k] for t in common]) for m in members]
        )
        tgt = np.stack([truth[t + k + 1] for t in common])  # (N, C, H, W)
        ok = np.isfinite(stack).all(axis=(0, 2, 3, 4))
        if ok.sum() == 0:
            w_out[k, :, :-1] = 1.0 / M  # fallback: equal weights
            continue
        stack, tgt = stack[:, ok], tgt[ok]
        # Latitude-weighted least squares: scaling both sides by sqrt(w) makes
        # ordinary LS minimize the latitude-weighted residual. The intercept
        # column gets the same sqrt(w) scaling (it is a constant 1 before
        # scaling), so its coefficient is exactly the constant offset that
        # BlendForecaster adds back.
        intercept = np.broadcast_to(sqrt_w, tgt.shape[-2:])  # (H, W)
        intercept = np.broadcast_to(intercept, tgt[:, 0].shape).ravel()
        for c in range(C):
            X = (stack[:, :, c] * sqrt_w).reshape(M, -1)  # (M, N*H*W)
            y = (tgt[:, c] * sqrt_w).ravel()
            A = np.concatenate([X, intercept[None]]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            w_out[k, c] = coef
    return w_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=["graphcast", "pangu", "hres"])
    p.add_argument("--leads", type=int, default=20)
    p.add_argument("--equal-weights", action="store_true",
                   help="plain member average: zero fitted parameters, so members "
                        "without a 2018 fitting set (e.g. gencast_mean) can join")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = DataConfig()
    lat = load_statics(cfg)["latitude"]
    lat_w = latitude_weights(lat)
    n_members = len(args.members)

    if args.equal_weights:
        C = 8
        weights = np.zeros((args.leads, C, n_members + 1), dtype=np.float32)
        weights[:, :, :-1] = 1.0 / n_members
        prefix = "avg_"
    else:
        print(f"fitting blend weights on 2018: {args.members}")
        weights = fit_weights(cfg, args.members, 2018, args.leads, lat_w)
        np.save(ARTIFACTS / "checkpoints" / "blend_weights.npy", weights)
        prefix = "blend_"

    members_2020 = [
        CompetitorForecaster(cfg, m if m.endswith("2020") else f"{m}_2020", 2020)
        for m in args.members
    ]
    fc = BlendForecaster(
        members_2020, weights, args.name or prefix + "+".join(args.members)
    )
    truth = np.asarray(load_years(cfg, [2020]), dtype=np.float32)
    times = year_range_times((2020, 2020))
    clim = np.asarray(load_climatology(CLIM_PATH))
    df = evaluate_forecaster(fc, truth, times, clim, lat_w, K=args.leads)
    df["split"] = "test"
    out = ARTIFACTS / "results" / f"{fc.name}_test.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    show = df[df.variable.isin(["u10", "v10", "wind_speed", "z500", "t2m"])]
    print(show[show.lead_h.isin([24, 72, 120])]
          .pivot_table(index="variable", columns="lead_h", values="rmse").round(3))


if __name__ == "__main__":
    main()

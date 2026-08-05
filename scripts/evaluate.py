"""Evaluate a forecaster over the 2020 test year -> artifacts/results/<name>.csv.

Usage:
    python scripts/evaluate.py --model persistence
    python scripts/evaluate.py --model climatology
    python scripts/evaluate.py --model linear
    python scripts/evaluate.py --ckpt artifacts/checkpoints/unet/best.pt
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import torch

from windml.config import ARTIFACTS, CHANNELS, DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.climatology import load_climatology
from windml.data.dataset import Era5Dataset, year_range_times
from windml.data.normalization import Normalizer, load_stats
from windml.eval.baselines import fit_linear
from windml.eval.forecasters import (
    ClimatologyForecaster,
    ModelForecaster,
    PersistenceForecaster,
)
from windml.eval.rollout import evaluate_forecaster
from windml.utils.grid import latitude_weights

CLIM_PATH = ARTIFACTS / "climatology" / "clim_train.npy"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, choices=["persistence", "climatology", "linear"])
    p.add_argument("--competitor", default=None, help="e.g. graphcast_2020, hres_2020")
    p.add_argument("--corrector-ckpt", default=None,
                   help="apply a trained corrector on top of --competitor")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--name", default=None, help="output name override")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--leads", type=int, default=20)
    p.add_argument("--init-stride", type=int, default=2)
    p.add_argument("--check", action="store_true", help="run sanity gates and exit nonzero on failure")
    args = p.parse_args()

    cfg = DataConfig()
    span = cfg.test_years if args.split == "test" else cfg.val_years
    truth = np.asarray(load_years(cfg, list(range(span[0], span[1] + 1))), dtype=np.float32)
    times = year_range_times(span)
    clim = np.asarray(load_climatology(CLIM_PATH))
    stats = load_stats(cfg.stats_path)
    norm = Normalizer(stats)
    lat = load_statics(cfg)["latitude"]
    lat_w = latitude_weights(lat)

    if args.ckpt:
        from windml.models import build_model  # registry import

        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        two_frame = payload.get("two_frame", True)
        name = args.name or payload.get("run_name", "model")
        direct_h = payload.get("direct_lead_h")
        # A model trained on the multi-level set needs its own cache and stats.
        # Build the dataset FIRST: only it knows the input-channel count, which
        # differs per variable set (49 for 'levels' vs 25 for 'core').
        mcfg = DataConfig(variable_set=payload.get("variable_set", "core"))
        mnorm = Normalizer(load_stats(mcfg.stats_path))
        ds = Era5Dataset(
            mcfg, span, mnorm, rollout_steps=1, two_frame=two_frame,
            direct_steps=(direct_h // 6) if direct_h else None,
        )
        model = build_model(
            payload["model_name"],
            in_channels=ds.n_input_channels,
            out_channels=len(mcfg.channels),
            **payload.get("model_params", {}),
        )
        model.load_state_dict(payload["state_dict"])
        if direct_h:
            # one-shot model: scores only its own lead, everything else NaN
            from windml.eval.forecasters import DirectForecaster

            fc = DirectForecaster(model, ds, name)
        else:
            fc = ModelForecaster(model, ds, mnorm, name)
    elif args.competitor:
        from windml.data.competitors import CompetitorForecaster

        year = int(args.competitor.rsplit("_", 1)[-1])
        fc = CompetitorForecaster(
            cfg, args.competitor, year, display_name=args.name or args.competitor
        )
        if args.corrector_ckpt:
            from windml.train.corrector import (
                CorrectedForecaster,
                CorrectorDataset,
                load_corrector,
            )

            payload = torch.load(args.corrector_ckpt, map_location="cpu", weights_only=False)
            corr_model = load_corrector(payload)
            corr_ds = CorrectorDataset(cfg, args.competitor, year, norm)
            fc = CorrectedForecaster(
                fc, corr_model, corr_ds, args.name or f"{args.competitor}+corr"
            )
    elif args.model == "persistence":
        fc = PersistenceForecaster(truth)
    elif args.model == "climatology":
        fc = ClimatologyForecaster(clim, times)
    elif args.model == "linear":
        train_ds = Era5Dataset(cfg, cfg.train_years, norm, rollout_steps=1, two_frame=False)
        model = fit_linear(train_ds)
        ds = Era5Dataset(cfg, span, norm, rollout_steps=1, two_frame=False)
        fc = ModelForecaster(model, ds, norm, "linear")
    else:
        raise SystemExit("pass --model or --ckpt")

    df = evaluate_forecaster(
        fc, truth, times, clim, lat_w, K=args.leads, init_stride=args.init_stride
    )
    df["git_sha"] = git_sha()
    df["split"] = args.split

    out = ARTIFACTS / "results" / f"{args.name or fc.name}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    show = df[df.variable.isin(["u10", "v10", "wind_speed", "z500", "t2m"])]
    show = show[show.lead_h.isin([24, 72, 120])]
    print(show.pivot_table(index="variable", columns="lead_h", values="rmse").round(3))

    if args.check:
        pers = df if fc.name == "persistence" else None
        ok = True
        if fc.name == "persistence":
            u10 = pers[pers.variable == "u10"].sort_values("lead_h")
            ok &= bool(u10.acc.is_monotonic_decreasing)
        if fc.name == "climatology":
            ws = df[df.variable == "wind_speed"]
            ok &= float(ws.rmse.std()) < 0.05 * float(ws.rmse.mean())
        if not ok:
            raise SystemExit("sanity checks FAILED")
        print("sanity checks passed")


if __name__ == "__main__":
    main()

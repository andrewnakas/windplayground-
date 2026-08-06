"""Probabilistic scoring: dress a deterministic forecaster and score CRPS.

The spread is calibrated on 2018 (a year not used for testing) and applied to
2020, so the calibration is fully out-of-sample.

Usage:
    python scripts/evaluate_ensemble.py --competitor graphcast_2020 \
        --corrector-ckpt artifacts/checkpoints/graphcast_corrector/best.pt
    python scripts/evaluate_ensemble.py --competitor graphcast_2020
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from windml.config import ARTIFACTS, DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.climatology import load_climatology
from windml.data.competitors import CompetitorForecaster
from windml.data.dataset import year_range_times
from windml.data.normalization import Normalizer, load_stats
from windml.eval.ensemble import DressedEnsemble, evaluate_ensemble, spread_from_scores
from windml.eval.rollout import evaluate_forecaster
from windml.utils.grid import latitude_weights

CLIM_PATH = ARTIFACTS / "climatology" / "clim_train.npy"
STATS_PATH = ARTIFACTS / "data" / "stats.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--competitor", required=True, help="e.g. graphcast_2020")
    p.add_argument("--corrector-ckpt", default=None)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--leads", type=int, default=20)
    p.add_argument("--init-stride", type=int, default=8,
                   help="CRPS is O(M^2); a coarser init stride keeps runtime sane")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = DataConfig()
    lat_w = latitude_weights(load_statics(cfg)["latitude"])
    clim = np.asarray(load_climatology(CLIM_PATH))
    norm = Normalizer(load_stats(STATS_PATH))

    def build(year: int):
        """Forecaster for `year`, optionally wrapped in the trained corrector."""
        base_name = args.competitor.rsplit("_", 1)[0] + f"_{year}"
        fc = CompetitorForecaster(cfg, base_name, year, display_name=base_name)
        if args.corrector_ckpt:
            from windml.train.corrector import (
                CorrectedForecaster,
                CorrectorDataset,
                load_corrector,
            )

            payload = torch.load(args.corrector_ckpt, map_location="cpu", weights_only=False)
            model = load_corrector(payload)
            ds = CorrectorDataset(cfg, base_name, year, norm)
            fc = CorrectedForecaster(fc, model, ds, f"{base_name}+corr")
        return fc

    # calibrate the dressing spread on 2018
    print("calibrating spread on 2018 ...")
    fc_2018 = build(2018)
    truth18 = np.asarray(load_years(cfg, [2018]), dtype=np.float32)
    times18 = year_range_times((2018, 2018))
    scores18 = evaluate_forecaster(
        fc_2018, truth18, times18, clim, lat_w, K=args.leads, init_stride=8
    )
    spread = spread_from_scores(scores18, args.leads)

    # score the dressed ensemble on 2020
    fc_2020 = build(2020)
    name = args.name or f"{fc_2020.name}+dressed{args.members}"
    ens = DressedEnsemble(fc_2020, spread, n_members=args.members)
    truth20 = np.asarray(load_years(cfg, [2020]), dtype=np.float32)
    times20 = year_range_times((2020, 2020))
    df = evaluate_ensemble(
        ens, truth20, times20, clim, lat_w, K=args.leads,
        init_stride=args.init_stride, name=name,
    )
    out = ARTIFACTS / "results" / f"{name}_crps.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    show = df[df.variable.isin(["u10", "v10", "wind_speed"])]
    print(show[show.lead_h.isin([24, 72, 120])]
          .pivot_table(index="variable", columns="lead_h", values="crps").round(3))


if __name__ == "__main__":
    main()

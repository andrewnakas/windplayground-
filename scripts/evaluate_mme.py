"""Score the multi-model ensemble (and single models) with fair CRPS.

Treating the published frontier forecasts as ensemble members needs no
fitting, so GenCast (2020-only) can take part. A single deterministic model
run through the same code gives CRPS == MAE, which is the reference row.

Usage:
    python scripts/evaluate_mme.py --members graphcast pangu hres gencast_mean_2020
    python scripts/evaluate_mme.py --members graphcast --name graphcast_det
"""
from __future__ import annotations

import argparse

import numpy as np

from windml.config import ARTIFACTS, DataConfig
from windml.data.build_cache import load_statics, load_years
from windml.data.climatology import load_climatology
from windml.data.competitors import CompetitorForecaster
from windml.data.dataset import year_range_times
from windml.eval.ensemble import MultiModelEnsemble, evaluate_ensemble
from windml.utils.grid import latitude_weights

CLIM_PATH = ARTIFACTS / "climatology" / "clim_train.npy"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--leads", type=int, default=20)
    p.add_argument("--init-stride", type=int, default=8)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = DataConfig()
    lat_w = latitude_weights(load_statics(cfg)["latitude"])
    clim = np.asarray(load_climatology(CLIM_PATH))
    members = [
        CompetitorForecaster(cfg, m if m.endswith("2020") else f"{m}_2020", 2020)
        for m in args.members
    ]
    name = args.name or ("mme_" + "+".join(args.members))
    ens = MultiModelEnsemble(members, name)

    truth = np.asarray(load_years(cfg, [2020]), dtype=np.float32)
    times = year_range_times((2020, 2020))
    df = evaluate_ensemble(
        ens, truth, times, clim, lat_w, K=args.leads,
        init_stride=args.init_stride, name=name,
    )
    out = ARTIFACTS / "results" / f"{name}_crps.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    show = df[df.variable.isin(["u10", "wind_speed"]) & df.lead_h.isin([24, 72, 120])]
    print(show.pivot_table(index="variable", columns="lead_h",
                           values=["crps", "spread_skill"]).round(3))


if __name__ == "__main__":
    main()

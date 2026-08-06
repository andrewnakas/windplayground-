"""Train the GraphCast corrector on 2018 forecasts; evaluate on 2020.

Usage:
    python scripts/train_corrector.py [--competitor graphcast] [--steps 6000]
        [--wind-weight 2.0] [--time-budget-hours 2.0]
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from windml.config import ARTIFACTS, DataConfig
from windml.data.build_cache import load_statics
from windml.data.normalization import Normalizer, load_stats
from windml.models import build_model
from windml.train.corrector import CorrectorDataset
from windml.train.losses import LatWeightedMSE, make_channel_weights
from windml.train.trainer import cosine_lr
from windml.utils.grid import latitude_weights

STATS_PATH = ARTIFACTS / "data" / "stats.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--competitor", default="graphcast")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--wind-weight", type=float, default=2.0)
    p.add_argument("--time-budget-hours", type=float, default=2.0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    cfg = DataConfig()
    norm = Normalizer(load_stats(STATS_PATH))
    train_ds = CorrectorDataset(cfg, f"{args.competitor}_2018", 2018, norm)
    print(f"corrector train pairs: {len(train_ds)}")

    model = build_model("unet", in_channels=train_ds.n_input_channels, width=args.width)
    # zero-init the output head so training starts from the identity correction
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"corrector params: {n_params/1e6:.2f}M")

    lat_w = latitude_weights(load_statics(cfg)["latitude"])
    wind_overrides = {
        "u10": args.wind_weight, "v10": args.wind_weight,
        "u850": args.wind_weight / 2 + 0.5, "v850": args.wind_weight / 2 + 0.5,
    }
    loss_fn = LatWeightedMSE(lat_w, make_channel_weights(wind_overrides))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    run_name = args.run_name or f"{args.competitor}_corrector"
    out_dir = ARTIFACTS / "checkpoints" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    step, t0 = 0, time.time()
    model.train()
    while step < args.steps:
        for x, y in loader:
            lr = cosine_lr(step, args.steps, args.lr, warmup=300)
            for g in opt.param_groups:
                g["lr"] = lr
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0:
                print(f"step {step:5d} lr {lr:.2e} loss {float(loss):.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            if step >= args.steps or time.time() - t0 > args.time_budget_hours * 3600:
                step = max(step, args.steps)
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "unet",
            "model_params": {"width": args.width},
            "in_channels": train_ds.n_input_channels,
            "competitor": args.competitor,
            "run_name": run_name,
        },
        out_dir / "best.pt",
    )
    print(f"saved {out_dir / 'best.pt'} after {step} steps")


if __name__ == "__main__":
    main()

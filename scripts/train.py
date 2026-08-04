"""Train a model from a YAML config.

Usage:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/vit.yaml --benchmark 30
    python scripts/train.py --config configs/unet.yaml --finetune-rollout 2 \
        --init-ckpt artifacts/checkpoints/unet/best.pt
"""
from __future__ import annotations

import argparse

import torch

from windml.config import ARTIFACTS, Config, DataConfig
from windml.data.build_cache import load_statics
from windml.data.dataset import Era5Dataset
from windml.data.normalization import Normalizer, load_stats
from windml.models import build_model
from windml.train.trainer import Trainer
from windml.utils.grid import latitude_weights

STATS_PATH = ARTIFACTS / "data" / "stats.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", type=int, default=0, help="time N steps and exit")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--time-budget-hours", type=float, default=None)
    p.add_argument("--finetune-rollout", type=int, default=None, help="override K")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.time_budget_hours is not None:
        cfg.train.time_budget_hours = args.time_budget_hours
    if args.finetune_rollout is not None:
        cfg.train.rollout_steps = args.finetune_rollout
    if args.run_name:
        cfg.run_name = args.run_name

    torch.set_num_threads(4)
    dcfg = cfg.data if isinstance(cfg.data, DataConfig) else DataConfig()
    stats = load_stats(STATS_PATH)
    norm = Normalizer(stats)
    train_ds = Era5Dataset(
        dcfg, dcfg.train_years, norm,
        rollout_steps=cfg.train.rollout_steps, two_frame=cfg.model.two_frame,
    )
    val_ds = Era5Dataset(
        dcfg, dcfg.val_years, norm,
        rollout_steps=cfg.train.rollout_steps, two_frame=cfg.model.two_frame,
    )
    train_ds.norm = norm
    val_ds.norm = norm

    model = build_model(cfg.model.name, in_channels=train_ds.n_input_channels,
                        **cfg.model.params)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.run_name}: {cfg.model.name} with {n_params/1e6:.2f}M params, "
          f"K={cfg.train.rollout_steps}")

    if args.init_ckpt:
        payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print(f"initialized from {args.init_ckpt} (step {payload.get('step')})")

    out_dir = ARTIFACTS / "checkpoints" / cfg.run_name
    trainer = Trainer(cfg, model, train_ds, val_ds, latitude_weights(
        load_statics(dcfg)["latitude"]), out_dir)
    result = trainer.train(benchmark_steps=args.benchmark)
    print(result)


if __name__ == "__main__":
    main()

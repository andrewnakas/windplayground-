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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", type=int, default=0, help="time N steps and exit")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--time-budget-hours", type=float, default=None)
    p.add_argument("--finetune-rollout", type=int, default=None, help="override K")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--direct-lead-h", type=int, default=None,
               help="train a one-shot model for this lead instead of 6h rollout")
    p.add_argument("--auto-resume", action="store_true",
                   help="continue from artifacts/checkpoints/<run>/last.pt if present")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.time_budget_hours is not None:
        cfg.train.time_budget_hours = args.time_budget_hours
    if args.direct_lead_h is not None:
        cfg.train.direct_lead_h = args.direct_lead_h
    if args.finetune_rollout is not None:
        cfg.train.rollout_steps = args.finetune_rollout
    if args.run_name:
        cfg.run_name = args.run_name

    torch.set_num_threads(4)
    dcfg = cfg.data if isinstance(cfg.data, DataConfig) else DataConfig()

    # Cheap early exit BEFORE touching the data. Loading the training array is
    # minutes (8.7 GB for the multi-level set), and the queue re-enters every
    # finished stage on each restart -- so checking the saved step count first
    # turns a 12-minute no-op into a 1-second one.
    out_dir = ARTIFACTS / "checkpoints" / cfg.run_name
    if args.auto_resume and (out_dir / "last.pt").exists():
        done = int(torch.load(out_dir / "last.pt", map_location="cpu",
                              weights_only=False).get("step", 0))
        if done >= cfg.train.max_steps:
            print(f"already at {done}/{cfg.train.max_steps} steps; nothing to do")
            return

    stats = load_stats(dcfg.stats_path)
    norm = Normalizer(stats)
    direct_steps = (cfg.train.direct_lead_h // 6) if cfg.train.direct_lead_h else None
    train_ds = Era5Dataset(
        dcfg, dcfg.train_years, norm, rollout_steps=cfg.train.rollout_steps,
        two_frame=cfg.model.two_frame, direct_steps=direct_steps,
        n_frames=cfg.model.n_frames,
    )
    val_ds = Era5Dataset(
        dcfg, dcfg.val_years, norm, rollout_steps=cfg.train.rollout_steps,
        two_frame=cfg.model.two_frame, direct_steps=direct_steps,
        n_frames=cfg.model.n_frames,
    )
    train_ds.norm = norm
    val_ds.norm = norm

    model = build_model(cfg.model.name, in_channels=train_ds.n_input_channels,
                        out_channels=len(dcfg.target_channels), **cfg.model.params)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.run_name}: {cfg.model.name} with {n_params/1e6:.2f}M params, "
          f"K={cfg.train.rollout_steps}")

    if args.init_ckpt:
        payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        src_set = payload.get("variable_set", dcfg.variable_set)
        state = payload["state_dict"]

        if src_set == dcfg.variable_set:
            model.load_state_dict(state)
        else:
            # Pretrain -> fine-tune across different variable sets. CMIP6 has
            # no 2m temperature and no precipitation, so a pretrained model is
            # narrower at BOTH ends: 111 inputs against 117, and 2 outputs
            # (z500, t850) against 3. A plain load_state_dict just raises here.
            #
            # Both ends are grown with zeros, which makes the transfer exact
            # rather than approximate: a conv sums over its input channels, so
            # a zero column contributes nothing, and a zero output row predicts
            # a zero residual -- "no change" -- which is the right neutral
            # start for a variable that was never supervised.
            from windml.config import active_variables
            from windml.models.grow import (
                channel_index_map,
                grow_input_channels,
                grow_output_channels,
            )

            src_channels = [v["short"] for v in active_variables(src_set)]
            keep = channel_index_map(
                src_channels, dcfg.channels,
                frames=train_ds.n_frames, n_static=train_ds.static.shape[0],
            )
            state = grow_output_channels(model, state)
            grow_input_channels(model, state, keep=keep)
            print(f"grew {src_set} -> {dcfg.variable_set}: "
                  f"{len(src_channels)}->{len(dcfg.channels)} fields, "
                  f"{len(keep)}->{train_ds.n_input_channels} inputs")
        print(f"initialized from {args.init_ckpt} (step {payload.get('step')})")

    trainer = Trainer(cfg, model, train_ds, val_ds, latitude_weights(
        load_statics(dcfg)["latitude"]), out_dir)
    start_step, best_val = trainer.load_resume_state() if args.auto_resume else (0, float("inf"))
    if start_step >= cfg.train.max_steps:
        print(f"already at {start_step}/{cfg.train.max_steps} steps; nothing to do")
        return
    result = trainer.train(
        benchmark_steps=args.benchmark, start_step=start_step, best_val=best_val
    )
    print(result)


if __name__ == "__main__":
    main()

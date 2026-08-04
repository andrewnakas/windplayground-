"""Training loop: AdamW + warmup/cosine, K-step rollout loss, time budgeting.

The model always predicts the normalized 6h residual. For K>1, the predicted
state is fed back autoregressively (input reassembled in normalized space) and
gradients flow through the whole chain (pushforward training).
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from windml.config import Config
from windml.data.dataset import Era5Dataset
from windml.train.losses import LatWeightedMSE


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: torch.nn.Module,
        train_ds: Era5Dataset,
        val_ds: Era5Dataset,
        lat_weights: np.ndarray,
        out_dir: str | Path,
    ):
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        from windml.train.losses import make_channel_weights

        self.loss_fn = LatWeightedMSE(
            lat_weights, make_channel_weights(cfg.train.channel_loss_weights)
        ).to(self.device)
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.K = cfg.train.rollout_steps
        self.diff_std_t = torch.from_numpy(
            train_ds.norm.diff_std.astype(np.float32)
        ).to(self.device)
        self.mean_t = torch.from_numpy(train_ds.norm.mean.astype(np.float32)).to(self.device)
        self.std_t = torch.from_numpy(train_ds.norm.std.astype(np.float32)).to(self.device)

    def _rollout_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: (B, F, H, W) input; y: (B, K, C, H, W) normalized residual targets."""
        C = y.shape[2]
        loss = torch.zeros((), device=self.device)
        state_norm = x[:, :C]  # normalized current state
        prev_norm = x[:, C : 2 * C] if self.train_ds.two_frame else None
        extras = x[:, 2 * C :] if self.train_ds.two_frame else x[:, C:]
        inp = x
        for k in range(self.K):
            pred = self.model(inp)
            loss = loss + self.loss_fn(pred, y[:, k])
            if k + 1 == self.K:
                break
            # advance the normalized state: x_{t+1} = x_t + resid, in physical
            # units; equivalently in normalized space via std ratios
            resid_phys = pred * self.diff_std_t
            state_phys = state_norm * self.std_t + self.mean_t + resid_phys
            new_norm = (state_phys - self.mean_t) / self.std_t
            prev_norm = state_norm
            state_norm = new_norm
            frames = [state_norm] + ([prev_norm] if self.train_ds.two_frame else [])
            # time features stale by 6h per step; acceptable at K<=4 (dominant
            # extras are static fields)
            inp = torch.cat(frames + [extras], dim=1)
        return loss / self.K

    @torch.no_grad()
    def validate(self, max_batches: int = 40) -> float:
        self.model.eval()
        loader = DataLoader(self.val_ds, batch_size=self.cfg.train.batch_size, shuffle=False)
        total, n = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            total += float(self._rollout_loss(x, y))
            n += 1
        self.model.train()
        return total / max(n, 1)

    def save(self, name: str, step: int, val_loss: float, best_val: float | None = None) -> None:
        # optimizer moments (2x the model size) are only needed to resume, so
        # they go in last.pt and not in the best.pt we keep as the artifact
        payload_opt = {"opt_state": self.opt.state_dict()} if name == "last.pt" else {}
        torch.save(
            {
                **payload_opt,
                "state_dict": self.model.state_dict(),
                "model_name": self.cfg.model.name,
                "model_params": self.cfg.model.params,
                "two_frame": self.cfg.model.two_frame,
                "run_name": self.cfg.run_name,
                "step": step,
                "val_loss": val_loss,
                "best_val": best_val if best_val is not None else val_loss,
                "config_hash": self.cfg.hash(),
            },
            self.out_dir / name,
        )

    def load_resume_state(self) -> tuple[int, float]:
        """Restore model/optimizer/step from last.pt so an interrupted run
        continues exactly where it stopped (LR schedule included).

        Returns (start_step, best_val); (0, inf) when there is nothing to resume.
        """
        path = self.out_dir / "last.pt"
        if not path.exists():
            return 0, float("inf")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        if "opt_state" in payload:
            self.opt.load_state_dict(payload["opt_state"])
        step = int(payload.get("step", 0))
        best = float(payload.get("best_val", payload.get("val_loss", float("inf"))))
        print(f"auto-resume from {path} at step {step} (best val {best:.4f})")
        return step, best

    def train(self, benchmark_steps: int = 0, start_step: int = 0,
              best_val: float = float("inf")) -> dict:
        cfg = self.cfg.train
        torch.manual_seed(cfg.seed + start_step)
        np.random.seed(cfg.seed + start_step)
        loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        log_path = self.out_dir / "log.csv"
        log_f = open(log_path, "a", newline="")
        logger = csv.writer(log_f)
        if log_path.stat().st_size == 0:
            logger.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_s"])

        max_steps = cfg.max_steps
        step = start_step
        t0 = time.time()
        self.model.train()
        done = False
        while not done:
            for x, y in loader:
                lr = cosine_lr(step, max_steps, cfg.lr, cfg.warmup_steps)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                x, y = x.to(self.device), y.to(self.device)
                loss = self._rollout_loss(x, y)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                step += 1

                if benchmark_steps and step >= benchmark_steps:
                    dt = (time.time() - t0) / step
                    log_f.close()
                    return {"steps_per_s": 1 / dt, "s_per_step": dt}

                elapsed = time.time() - t0
                if cfg.time_budget_hours and elapsed > cfg.time_budget_hours * 3600:
                    print(f"time budget reached at step {step}")
                    done = True
                if step % cfg.val_every == 0 or done or step >= max_steps:
                    val_loss = self.validate()
                    logger.writerow([step, f"{lr:.2e}", f"{float(loss):.4f}",
                                     f"{val_loss:.4f}", f"{elapsed:.0f}"])
                    log_f.flush()
                    print(f"step {step:6d} lr {lr:.2e} train {float(loss):.4f} "
                          f"val {val_loss:.4f} [{elapsed:.0f}s]")
                    if val_loss < best_val:
                        best_val = val_loss
                        self.save("best.pt", step, val_loss, best_val)
                    # checkpoint resume state every val interval: a container
                    # restart then costs at most one interval of progress
                    self.save("last.pt", step, val_loss, best_val)
                if step >= max_steps:
                    done = True
                if done:
                    break
        self.save("last.pt", step, best_val, best_val)
        log_f.close()
        return {"steps": step, "best_val": best_val, "elapsed_s": time.time() - t0}

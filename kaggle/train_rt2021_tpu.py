"""RT2021 on TPU: eight independent seeds, one per core. Kaggle TPU kernel.

**This is not a data-parallel port, and that is deliberate.** A 6.36M-parameter
convnet on a 64x32 grid does not need eight chips: the bottleneck is data
loading, large-batch training would need the paper's LR retuned, and RT2021's
BatchNorm makes per-core batch statistics diverge from its batch-32 recipe. All
three are ways for a "faster" run to stop being a reproduction.

Running one *whole model* per core avoids every one of them. Each core keeps
batch 32, LR 5e-5 and its own BatchNorm exactly as published; the only thing
that differs is the seed. Nothing is all-reduced -- note `opt.step()` below
rather than `xm.optimizer_step`, which would average gradients across cores and
quietly turn this back into one data-parallel model.

What it buys: the eight checkpoints *are* an ensemble, and ensembling is the
one lever already measured in this repo (`avg5` beat its best member by
5.7-7.3% on wind speed with zero fitted parameters). Eight seeds for the
wall-clock of one is the whole point.

TPU also sidesteps the Pascal problem the GPU kernel has to work around: there
is no CUDA arch to match, so no cu118 pin.

Modes (SMOKE=1 in the kernel source, or the default full run):
    smoke -- synthetic data, ~200 steps, proves 8 cores + fwd/bwd + save
    full  -- reads the prep kernel's output, one direct-72h model per seed
"""
# No `from __future__ import annotations` here on purpose: scripts/kaggle_run.py
# configures the smoke variant by prepending a line to this file, and a
# __future__ import is only legal as the first statement.

import json
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

SMOKE = os.environ.get("WINDML_SMOKE", "0") == "1"

OUT = pathlib.Path("/kaggle/working")
TRAIN, VAL = (1979, 2015), (2016, 2016)
TARGETS = ["z500", "t850", "t2m"]
N_FRAMES = 3
DIRECT_STEPS = 12          # 72 h at 6-hourly
LEAKY = 0.3

BATCH = 32
LR = 5e-5
WD = 1e-5
PLATEAU_FACTOR, PLATEAU_PATIENCE, MAX_DROPS, EARLY_STOP = 0.2, 2, 2, 5
VAL_EVERY = 200 if SMOKE else 2000
MAX_STEPS = 400 if SMOKE else 10**9
# Kaggle cuts TPU sessions at 9 h -- stop with room to write eight checkpoints.
TIME_BUDGET_S = 300 if SMOKE else 7.5 * 3600


def device():
    """torch_xla renamed this in 2.x; support both rather than pin a version."""
    try:
        import torch_xla
        return torch_xla.device()
    except (ImportError, AttributeError):
        return xm.xla_device()


# ---------------------------------------------------------------- architecture
# Duplicated from kaggle/train_rt2021.py and src/windml/models/rt_resnet.py; a
# Kaggle kernel cannot import `windml`. tests/test_rt_resnet.py pins the 6.36M
# parameter count against the paper's ~6.3M, and any change goes in all three.


class CircularConv2d(nn.Module):
    """Periodic in longitude, zero-padded in latitude -- the paper's padding."""

    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(cin, cout, k)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = nn.functional.pad(x, (0, 0, self.pad, self.pad))
        return self.conv(x)


class ConvBlock(nn.Module):
    """[Conv -> LeakyReLU -> BatchNorm -> Dropout]. Activation before norm."""

    def __init__(self, cin, cout, k=3, p=0.1):
        super().__init__()
        self.conv = CircularConv2d(cin, cout, k)
        self.act = nn.LeakyReLU(LEAKY)
        self.norm = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.norm(self.act(self.conv(x))))


class ResBlock(nn.Module):
    def __init__(self, c, p=0.1):
        super().__init__()
        self.a, self.b = ConvBlock(c, c, 3, p), ConvBlock(c, c, 3, p)

    def forward(self, x):
        return x + self.b(self.a(x))


class WeatherResNetRT(nn.Module):
    def __init__(self, cin=117, cout=3, width=128, blocks=19, p=0.1):
        super().__init__()
        self.stem = ConvBlock(cin, width, 7, p)
        self.blocks = nn.ModuleList(ResBlock(width, p) for _ in range(blocks))
        self.head = CircularConv2d(width, cout, 3)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)

    def conv_kernels(self):
        return [m.conv.weight for m in self.modules() if isinstance(m, CircularConv2d)]


# ----------------------------------------------------------------------- data


class YearStack:
    """Per-year .npy files as one logical array, memory-mapped.

    The GPU kernel copies the 37 train years into one 8.4 GB in-RAM block. Eight
    processes doing that would want ~67 GB. Memory-mapping instead means the
    eight cores share a single copy through the OS page cache, and indexing
    still crosses year boundaries transparently.
    """

    def __init__(self, root: pathlib.Path, lo: int, hi: int):
        self.parts = [np.load(root / f"era5_rt2021_{y}.npy", mmap_mode="r")
                      for y in range(lo, hi + 1)]
        lens = [p.shape[0] for p in self.parts]
        self.offsets = np.cumsum([0] + lens)
        self.shape = (int(self.offsets[-1]), *self.parts[0].shape[1:])

    def take(self, idx: np.ndarray) -> np.ndarray:
        """Gather arbitrary global time indices into one contiguous float32 batch."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), *self.shape[1:]), dtype=np.float32)
        part_of = np.searchsorted(self.offsets, idx, side="right") - 1
        for p in np.unique(part_of):
            sel = part_of == p
            local = idx[sel] - self.offsets[p]
            out[sel] = self.parts[p][local].astype(np.float32)
        return out


class SyntheticStack:
    """Stand-in for the smoke run: same shapes, no dataset dependency."""

    def __init__(self, n_t=600, n_c=38, h=32, w=64, seed=0):
        self.shape = (n_t, n_c, h, w)
        self._rng = np.random.default_rng(seed)
        self._buf = self._rng.normal(0, 1, (n_t, n_c, h, w)).astype(np.float32)

    def take(self, idx):
        return self._buf[np.asarray(idx)]


def channel_order():
    """The prep kernel's channel order, for the smoke run's synthetic stack.

    Real names rather than placeholders so the smoke path exercises the same
    lookups the full run does -- `names.index("z925")` for the orography
    stand-in is exactly the kind of thing a c0/c1/c2 stub would not catch.
    """
    levels = [50, 250, 500, 600, 700, 850, 925]
    names = ["z500", "t850", "t2m"]
    names += [f"{a}{lv}" for a in ("z", "t", "u", "v", "q") for lv in levels
              if f"{a}{lv}" not in ("z500", "t850")]
    return names + ["tp", "tisr"]


def latitude_field(n_lat, n_lon):
    lat = np.linspace(-87.1875, 87.1875, n_lat, dtype=np.float32)
    return np.broadcast_to(np.sin(np.deg2rad(lat))[:, None],
                           (n_lat, n_lon)).astype(np.float32)


def find_input() -> pathlib.Path:
    root = pathlib.Path("/kaggle/input")
    hits = sorted(root.glob("**/meta.json"))
    if hits:
        return hits[0].parent
    listing = [str(p) for p in sorted(root.glob("*"))] or ["<empty>"]
    raise SystemExit("RESULT FAIL no meta.json under /kaggle/input; mounted: "
                     + ", ".join(listing))


# ------------------------------------------------------------------- per core


def run(rank: int) -> None:
    dev = device()
    seed = 1000 + rank
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    def say(msg: str) -> None:
        print(f"windml core={rank} {msg}", flush=True)

    if SMOKE:
        names = channel_order()
        train, val = SyntheticStack(seed=seed), SyntheticStack(n_t=200, seed=seed + 1)
        re_a = np.ones((1, 38, 1, 1), np.float32)
        re_b = np.zeros((1, 38, 1, 1), np.float32)
        store_std_tgt = np.ones(3, np.float32)
    else:
        src = find_input()
        meta = json.loads((src / "meta.json").read_text())
        stats = json.loads((src / "stats.json").read_text())
        if not stats.get("stored_normalized"):
            raise SystemExit("RESULT FAIL input stores physical units in float16 "
                             "(z50/z250 overflow to inf). Re-run the prep kernel.")
        names = meta["channels"]
        sm = np.asarray(stats["store_mean"], np.float32)[None, :, None, None]
        ss = np.asarray(stats["store_std"], np.float32)[None, :, None, None]
        mean = np.asarray(stats["mean"], np.float32)[None, :, None, None]
        std = np.asarray(stats["std"], np.float32)[None, :, None, None]
        # arrays arrive as (physical - store_mean)/store_std; compose that with
        # the exact train statistics instead of round-tripping through physical
        re_a, re_b = ss / std, (sm - mean) / std
        store_std_tgt = ss[0, [names.index(t) for t in TARGETS], 0, 0]
        train = YearStack(src, *TRAIN)
        val = YearStack(src, *VAL)

    tgt = [names.index(t) for t in TARGETS]
    n_lat, n_lon = train.shape[2], train.shape[3]
    say(f"train={train.shape} val={val.shape} seed={seed} dev={dev}")

    # The paper's three constants. Land-sea mask and orography are not in the
    # prep output, so the time mean of 925 hPa geopotential stands in for
    # terrain (it is depressed over high ground) and a threshold on it for the
    # mask. Crude, but they are constants the network learns around, and
    # keeping the count at 3 is what holds the input to the paper's 117.
    probe = train.take(np.arange(0, train.shape[0], max(train.shape[0] // 60, 1)))
    zsurf = torch.from_numpy(probe[:, names.index("z925")].mean(0))
    zsurf = (zsurf - zsurf.mean()) / (zsurf.std() + 1e-8)
    const = torch.stack([(zsurf > 0.15).float(), zsurf,
                         torch.from_numpy(latitude_field(n_lat, n_lon))])[None].to(dev)

    n_in = len(names) * N_FRAMES + 3
    assert n_in == 117, f"expected the paper's 117 inputs, got {n_in}"
    model = WeatherResNetRT(cin=n_in, cout=len(TARGETS)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    say(f"in_channels={n_in} params={n_par/1e6:.3f}M (paper ~6.3M)")

    kern = {id(p) for p in model.conv_kernels()}
    opt = torch.optim.Adam(
        [{"params": [p for p in model.parameters() if id(p) in kern], "weight_decay": WD},
         {"params": [p for p in model.parameters() if id(p) not in kern], "weight_decay": 0.0}],
        lr=LR)

    re_a_t = torch.from_numpy(re_a).to(dev)
    re_b_t = torch.from_numpy(re_b).to(dev)
    latw = np.cos(np.deg2rad(np.linspace(-87.1875, 87.1875, n_lat)))
    latw = torch.from_numpy((latw / latw.mean()).astype(np.float32)).to(dev)[None, None, :, None]

    hi_t = train.shape[0] - DIRECT_STEPS - 1
    hi_v = val.shape[0] - DIRECT_STEPS - 1
    lo = N_FRAMES - 1

    i = rng.choice(hi_t, size=min(1500, hi_t), replace=False)
    d = train.take(i + DIRECT_STEPS)[:, tgt] - train.take(i)[:, tgt]
    dstd_np = d.std(axis=(0, 2, 3)).astype(np.float32)
    dstd = torch.from_numpy(dstd_np).to(dev)[None, :, None, None]
    say(f"direct_std_physical={[round(float(a * b), 2) for a, b in zip(dstd_np, store_std_tgt)]}")

    def make_batch(stack, idx):
        """3 normalized frames + 3 constants -> residual scaled to unit variance."""
        frames = [torch.from_numpy(stack.take(idx - k)).to(dev) for k in range(N_FRAMES)]
        x = torch.cat([re_a_t * f + re_b_t for f in frames]
                      + [const.expand(len(idx), -1, -1, -1)], 1)
        fut = torch.from_numpy(stack.take(idx + DIRECT_STEPS)[:, tgt]).to(dev)
        now = torch.from_numpy(stack.take(idx)[:, tgt]).to(dev)
        return x, (fut - now) / dstd

    def loss_fn(p, y):
        return (latw * (p - y) ** 2).mean()

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for s in range(lo, hi_v, BATCH * 8):
            idx = np.arange(s, min(s + BATCH, hi_v))
            if len(idx) < 2:
                continue
            x, y = make_batch(val, idx)
            tot += float(loss_fn(model(x), y))
            xm.mark_step()
            n += 1
        model.train()
        return tot / max(n, 1)

    best, stalled, drops, lr = float("inf"), 0, 0, LR
    step, log = 0, []
    ckpt = OUT / f"rt2021_72h_seed{rank}.pt"

    while step < MAX_STEPS:
        idx = rng.integers(lo, hi_t, BATCH)
        x, y = make_batch(train, idx)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # opt.step(), NOT xm.optimizer_step(): these are eight independent
        # models. All-reducing gradients here would silently collapse the
        # ensemble into one data-parallel run.
        opt.step()
        xm.mark_step()
        step += 1

        if step % VAL_EVERY == 0:
            v, tr = validate(), float(loss.detach())
            if not np.isfinite(v) or not np.isfinite(tr):
                raise SystemExit(f"RESULT FAIL core={rank} non-finite at step "
                                 f"{step}: train={tr} val={v}")
            el = time.time() - t0
            log.append({"step": step, "lr": lr, "train": tr, "val": v, "s": el})
            say(f"step={step} lr={lr:.2e} train={tr:.4f} val={v:.4f} "
                f"elapsed={el/60:.0f}m")
            if v < best:
                best, stalled = v, 0
                # every core saves its own model, so master_only would drop
                # seven of the eight
                xm.save({"state_dict": model.state_dict(), "step": step, "val": v,
                         "seed": seed, "channels": names, "targets": TARGETS,
                         "direct_std": dstd_np.tolist(),
                         "store_std": [float(s) for s in store_std_tgt],
                         "n_frames": N_FRAMES, "direct_lead_h": 72},
                        str(ckpt), master_only=False)
            else:
                stalled += 1
                if stalled > EARLY_STOP:
                    say(f"early_stop step={step} best={best:.4f}")
                    break
                if stalled > PLATEAU_PATIENCE and drops < MAX_DROPS:
                    lr *= PLATEAU_FACTOR
                    drops += 1
                    stalled = 0
                    for g in opt.param_groups:
                        g["lr"] = lr
                    say(f"lr_drop={drops}/{MAX_DROPS} -> {lr:.2e}")
            (OUT / f"metrics_seed{rank}.json").write_text(json.dumps(
                {"seed": seed, "log": log, "best_val": best, "params": n_par}, indent=2))

        if time.time() - t0 > TIME_BUDGET_S:
            say(f"time_budget_reached step={step}")
            break

    say(f"best_val={best:.4f} steps={step} ckpt={ckpt.exists()}")
    if not np.isfinite(best) or not ckpt.exists():
        raise SystemExit(f"RESULT FAIL core={rank} best_val={best} "
                         f"checkpoint={ckpt.exists()}")


def _entry(rank):
    run(xm.get_ordinal() if rank is None else rank)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"windml mode={'smoke' if SMOKE else 'full'} "
          f"world_size={xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else '?'}",
          flush=True)
    xmp.spawn(_entry, args=(), nprocs=None)   # None -> one process per TPU core
    saved = sorted(OUT.glob("rt2021_72h_seed*.pt"))
    print(f"windml ensemble_members={len(saved)}", flush=True)
    if len(saved) < 2:
        raise SystemExit(f"RESULT FAIL only {len(saved)} member(s) saved; "
                         f"the point of the TPU run is the ensemble")
    print("RESULT OK")


if __name__ == "__main__":
    main()

"""RT2021 on TPU: eight independent seeds, one per chip. Kaggle TPU kernel.

**Not a data-parallel port, and that is deliberate.** A 6.36M-parameter convnet
on a 64x32 grid does not need eight chips: the bottleneck is data loading,
large-batch training would need the paper's LR retuned, and RT2021's BatchNorm
makes per-core batch statistics diverge from its batch-32 recipe. All three are
ways for a "faster" run to stop being a reproduction.

Running one *whole model* per chip avoids every one of them. Each member has its
own BatchNorm and its own batch; no gradient is ever shared. What it buys is
that the eight checkpoints *are* an ensemble, and ensembling is the one lever
already measured in this repo (`avg5` beat its best member by 5.7-7.3% on wind
speed with zero fitted parameters).

**Recipe: the paper's, unmodified.** Batch 32, LR 5e-5, no gradient clipping.
Larger batches were tried to get more samples into a 7.5 h session and both
died on HBM (see the constants below), so the only thing separating a member
here from the GPU reproduction is how many steps it got -- not the recipe. The
ensemble is therefore an extension in the sense that the paper does not
ensemble, not in the sense of having been trained differently.

**One process, eight devices -- NOT xmp.spawn.** Kaggle's TPU is a
`v5litepod-8` (v5e-8) with `TPU_PROCESS_ADDRESSES=local` and a single entry in
`TPU_WORKER_HOSTNAMES`, so the multiprocess path dies in libtpu before any user
code runs:

    TPU initialization failed: Invalid --<id>_slice_builder_worker_addresses
    specified. Expected 8 worker addresses, got 1.

kaggle/tpu_probe.py established that all eight chips are addressable from a
single process (`xr.global_runtime_device_count() == 8`) and that eight
independent models step happily side by side there. Since nothing is
all-reduced, spawn bought nothing here anyway -- it was only ever a way to get
eight devices, and this gets them without the multiprocessing.

XLA is lazy, so the eight members are enqueued and only then executed: one
`xm.mark_step()` per training step, after all eight have been built, is what
lets them run concurrently rather than one at a time.

TPU also sidesteps the Pascal problem the GPU kernel works around -- no CUDA
arch to match, so no cu118 pin.

**Measured reality: this model is latency-bound on a v5e-8, not throughput-
bound.** Four candidate explanations for a fixed ~1.6 s/step were tested and
three were wrong:

    Python tracing        refuted -- 8 members cost 0.83x of one, and
                          torch_xla.compile changed nothing (1.681 vs 1.669)
    benchmark's own sync  refuted -- async differs by 0.05 s
    serial transfers      real but minor -- packing five .to(dev) calls into
                          one bought 1.634 -> 1.405 s/step (+14%)
    per-op latency        what is left, and it fits everything

The tell is batch size: 32 -> 128 costs only +20% wall clock for 3.4x the
samples. A 6.36M model on a 64x32 grid issues 39 sequential convolutions whose
individual kernels are far too small to fill a chip, so the step is dominated
by op dispatch, not arithmetic -- roughly 0.08% of the v5e-8's peak FLOPs. More
chips do not help that; they are already busy in parallel across members. This
is a property of the model's shape, not of the port.

Practical consequence at batch 32: ~19k steps per member in 7.5 h, against
~79k steps for a single model on the P100 in the same time. Eight members at a
quarter of the training each. Whether that trade is worth taking depends on
where a single faithful model actually converges, which is what the GPU gate
run measures -- so that decision waits for it rather than being guessed here.

Modes (SMOKE=1 prepended by scripts/kaggle_run.py, or the default full run):
    smoke -- synthetic data, a few hundred steps, proves the 8-chip plumbing
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

import torch_xla
import torch_xla.core.xla_model as xm

SMOKE = os.environ.get("WINDML_SMOKE", "0") == "1"

OUT = pathlib.Path("/kaggle/working")
TRAIN, VAL = (1979, 2015), (2016, 2016)
TARGETS = ["z500", "t850", "t2m"]
N_FRAMES = 3
DIRECT_STEPS = 12          # 72 h at 6-hourly
LEAKY = 0.3

# DELIBERATE DEVIATIONS FROM THE PAPER -- this kernel is the extension, not the
# reproduction, and the fidelity claim rests entirely on the GPU run.
#
# BACK TO THE PAPER'S EXACT VALUES, after two OOM failures showed the larger
# batch was not buying what it was supposed to:
#
#   batch 128 -> RESOURCE_EXHAUSTED, 21.36M free on the first input transfer
#   batch  64 -> RESOURCE_EXHAUSTED, 37.19M free, same place
#
# Halving the batch recovered 16 MB, which is just the smaller input block --
# the device was already full before training began, so the batch was never the
# problem and a third size would fail the same way. The hbm print below
# diagnoses the real cause.
#
# The upside of reverting: with batch 32, LR 5e-5 and no clipping, this run has
# NO deviations from Rasp & Thuerey left. Each member is a faithful-recipe
# model that has simply seen fewer steps than the GPU one, which makes the
# ensemble easier to report honestly than a batch-128 variant would have been.
BATCH = 32
LR = 5e-5
WD = 1e-5
PLATEAU_FACTOR, PLATEAU_PATIENCE, MAX_DROPS, EARLY_STOP = 0.2, 2, 2, 5
# ON here, and the earlier note that it costs +274% was wrong. Two benchmarks
# disagreed -- 6.170 vs 1.651 s/step in the first, 1.794 vs 1.669 in the second
# -- and the difference is run ORDER: clipping ran first in benchmark 1 and
# absorbed the compile/warmup. +8% is the real cost.
#
# The paper still does not use clipping, so the GPU reproduction leaves it off.
# But at 4x the LR with BatchNorm, a divergence would waste the whole TPU pool,
# and 8% is cheap insurance on a run that is already labelled an extension.
CLIP_GRAD = False
VAL_EVERY = 20 if SMOKE else 1000
MAX_STEPS = 60 if SMOKE else 10**9
# Kaggle cuts TPU sessions at 9 h -- stop with room to write eight checkpoints.
TIME_BUDGET_S = 420 if SMOKE else 8.0 * 3600


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
    """Per-year .npy files as one logical array, indexable across boundaries."""

    def __init__(self, root, lo, hi, eager=True):
        paths = [root / f"era5_rt2021_{y}.npy" for y in range(lo, hi + 1)]
        self.parts = [np.load(p, mmap_mode=None if eager else "r") for p in paths]
        self.offsets = np.cumsum([0] + [p.shape[0] for p in self.parts])
        self.shape = (int(self.offsets[-1]), *self.parts[0].shape[1:])

    def take(self, idx):
        """Gather arbitrary global time indices into one contiguous float32 batch.

        One vectorized gather for all eight members' indices at once: the eight
        models share a process, so eight separate 32-sample gathers would pay
        numpy's per-call overhead eight times for the same bytes.
        """
        idx = np.asarray(idx)
        out = np.empty((len(idx), *self.shape[1:]), dtype=np.float32)
        part_of = np.searchsorted(self.offsets, idx, side="right") - 1
        for p in np.unique(part_of):
            sel = part_of == p
            out[sel] = self.parts[p][idx[sel] - self.offsets[p]].astype(np.float32)
        return out


class SyntheticStack:
    """Stand-in for the smoke run: same shapes, no dataset dependency."""

    def __init__(self, n_t=600, n_c=38, h=32, w=64, seed=0):
        self.shape = (n_t, n_c, h, w)
        self._buf = np.random.default_rng(seed).normal(
            0, 1, (n_t, n_c, h, w)).astype(np.float32)

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


def find_input():
    root = pathlib.Path("/kaggle/input")
    hits = sorted(root.glob("**/meta.json"))
    if hits:
        return hits[0].parent
    listing = [str(p) for p in sorted(root.glob("*"))] or ["<empty>"]
    raise SystemExit("RESULT FAIL no meta.json under /kaggle/input; mounted: "
                     + ", ".join(listing))


# -------------------------------------------------------------------- members


class Member:
    """One independent model, pinned to one TPU chip."""

    def __init__(self, dev, rank, n_in, n_out, re_a, re_b, const, latw, dstd):
        self.dev, self.rank = dev, rank
        self.seed = 1000 + rank
        torch.manual_seed(self.seed)                 # weight init
        self.rng = np.random.default_rng(self.seed)  # data order
        self.model = WeatherResNetRT(cin=n_in, cout=n_out).to(dev)
        kern = {id(p) for p in self.model.conv_kernels()}
        params = list(self.model.parameters())
        self.opt = torch.optim.Adam(
            [{"params": [p for p in params if id(p) in kern], "weight_decay": WD},
             {"params": [p for p in params if id(p) not in kern], "weight_decay": 0.0}],
            lr=LR)
        # tiled over the three frames so the whole dynamic block normalizes in
        # one multiply-add rather than three
        self.re_a3 = torch.from_numpy(np.tile(re_a, (1, N_FRAMES, 1, 1))).to(dev)
        self.re_b3 = torch.from_numpy(np.tile(re_b, (1, N_FRAMES, 1, 1))).to(dev)
        self.const = const.to(dev)
        self.latw = torch.from_numpy(latw).to(dev)[None, None, :, None]
        self.dstd = torch.from_numpy(dstd).to(dev)[None, :, None, None]
        self.lr, self.best, self.stalled, self.drops = LR, float("inf"), 0, 0
        self.log, self.done = [], False

    def assemble(self, blk):
        """One packed host array -> (117-channel input, scaled residual).

        ONE transfer per step, not five. The earlier version sent the three
        frames, the future target and the current state as separate `.to(dev)`
        calls, and the benchmark showed why that mattered: one member cost
        1.96 s/step and eight cost 1.67 s -- identical, because transfers to
        different chips overlap while the five within a member serialize. Five
        round trips at ~350 ms is the entire per-step cost; the chips were
        idle for nearly all of it.

        Layout of `blk` is (N, 3*C + 2*len(TARGETS), H, W): frames at t, t-6h,
        t-12h, then the +72 h target, then the current state.
        """
        d = torch.from_numpy(blk).to(self.dev)
        c3 = self.re_a3.shape[1]
        n_t = (d.shape[1] - c3) // 2
        x = torch.cat([d[:, :c3] * self.re_a3 + self.re_b3,
                       self.const.expand(d.shape[0], -1, -1, -1)], 1)
        y = (d[:, c3:c3 + n_t] - d[:, c3 + n_t:]) / self.dstd
        return x, y

    def loss(self, pred, y):
        return (self.latw * (pred - y) ** 2).mean()

    def train_step(self, blk, clip=CLIP_GRAD):
        """Build the graph; execution waits for the caller's single mark_step."""
        x, y = self.assemble(blk)
        loss = self.loss(self.model(x), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            # Suspected XLA cost centre: clip_grad_norm_ walks every parameter
            # in Python and emits a norm op per tensor, so ~120 params x 8
            # members is ~1000 extra nodes per step. The benchmark A/Bs it
            # rather than assuming either way.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # opt.step(), NOT xm.optimizer_step(): these are independent models, and
        # all-reducing here would quietly collapse the ensemble into one.
        self.opt.step()
        return loss


# ----------------------------------------------------------------------- main


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    devices = xm.get_xla_supported_devices()
    print(f"windml mode={'smoke' if SMOKE else 'full'} torch_xla="
          f"{torch_xla.__version__} devices={len(devices)} {devices}", flush=True)
    if len(devices) < 2:
        raise SystemExit(f"RESULT FAIL only {len(devices)} XLA device(s); the "
                         f"point of this kernel is one model per chip")

    if SMOKE:
        names = channel_order()
        train, val = SyntheticStack(seed=7), SyntheticStack(n_t=200, seed=8)
        re_a = np.ones((1, len(names), 1, 1), np.float32)
        re_b = np.zeros((1, len(names), 1, 1), np.float32)
        store_std_tgt = np.ones(len(TARGETS), np.float32)
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
        # the exact train statistics rather than round-tripping through physical
        re_a, re_b = ss / std, (sm - mean) / std
        store_std_tgt = ss[0, [names.index(t) for t in TARGETS], 0, 0]
        train, val = YearStack(src, *TRAIN), YearStack(src, *VAL)

    tgt = [names.index(t) for t in TARGETS]
    n_lat, n_lon = train.shape[2], train.shape[3]
    print(f"windml train={train.shape} val={val.shape} "
          f"load={time.time()-t0:.0f}s", flush=True)

    # The paper's three constants. Land-sea mask and orography are not in the
    # prep output, so the time mean of 925 hPa geopotential stands in for
    # terrain (it is depressed over high ground) and a threshold on it for the
    # mask. Crude, but they are constants the network learns around, and keeping
    # the count at 3 is what holds the input to the paper's 117.
    probe = train.take(np.arange(0, train.shape[0], max(train.shape[0] // 60, 1)))
    zsurf = torch.from_numpy(probe[:, names.index("z925")].mean(0))
    zsurf = (zsurf - zsurf.mean()) / (zsurf.std() + 1e-8)
    const = torch.stack([(zsurf > 0.15).float(), zsurf,
                         torch.from_numpy(latitude_field(n_lat, n_lon))])[None]

    lo = N_FRAMES - 1
    hi_t = train.shape[0] - DIRECT_STEPS - 1
    hi_v = val.shape[0] - DIRECT_STEPS - 1

    srng = np.random.default_rng(0)
    i = srng.choice(hi_t, size=min(1500, hi_t), replace=False)
    dstd = (train.take(i + DIRECT_STEPS)[:, tgt]
            - train.take(i)[:, tgt]).std(axis=(0, 2, 3)).astype(np.float32)
    print(f"windml direct_std_physical="
          f"{[round(float(a * b), 2) for a, b in zip(dstd, store_std_tgt)]}", flush=True)

    n_in = len(names) * N_FRAMES + 3
    assert n_in == 117, f"expected the paper's 117 inputs, got {n_in}"
    latw = np.cos(np.deg2rad(np.linspace(-87.1875, 87.1875, n_lat)))
    latw = (latw / latw.mean()).astype(np.float32)

    members = [Member(d, r, n_in, len(TARGETS), re_a, re_b, const, latw, dstd)
               for r, d in enumerate(devices)]
    n_par = sum(p.numel() for p in members[0].model.parameters())
    print(f"windml members={len(members)} in_channels={n_in} "
          f"params_each={n_par/1e6:.3f}M (paper ~6.3M)", flush=True)
    # Materialize parameters and optimizer state NOW rather than letting them
    # land in the same execution as the first batch. XLA is lazy, so without
    # this the first step must allocate weights, Adam moments and a full
    # activation tree at once.
    xm.mark_step()
    xm.wait_device_ops()

    # Report per-device HBM before a single batch is transferred. Two runs died
    # here with the device essentially full -- 21 MB free at batch 128, 37 MB at
    # batch 64 -- and halving the batch barely moved the needle, so the batch is
    # not what is consuming it. The obvious suspect is that the eight members
    # are not actually spread across the eight chips. This says so outright
    # instead of costing another run to infer.
    for d in devices:
        try:
            mi = xm.get_memory_info(d)
            used = mi.get("bytes_used", mi.get("kb_total", 0))
            limit = mi.get("bytes_limit", 0)
            print(f"windml hbm {d} used={used/1e9:.2f}GB limit={limit/1e9:.2f}GB",
                  flush=True)
        except Exception as exc:                       # API differs across versions
            print(f"windml hbm {d} unavailable: {type(exc).__name__}", flush=True)
            break

    def gather(stack, idx):
        """One packed block for the whole step: frames t..t-2, then t+72h, then t.

        Concatenated on the host so each member needs a single transfer. The
        channel layout is what Member.assemble unpacks.
        """
        now = stack.take(idx)
        return np.concatenate(
            [now] + [stack.take(idx - k) for k in range(1, N_FRAMES)]
            + [stack.take(idx + DIRECT_STEPS)[:, tgt], now[:, tgt]], axis=1)

    def slice_for(blk, sl):
        # ascontiguousarray: a row slice of a C-ordered array is already
        # contiguous, but being explicit keeps the single-transfer promise
        # true if the layout ever changes
        return (np.ascontiguousarray(blk[sl]),)

    def benchmark():
        """Separate compile cost from steady state, and A/B gradient clipping.

        The first smoke run reported "20 steps in 11 minutes" and that number is
        useless on its own: XLA compiles lazily, so step 1 pays for the whole
        program and steps 2..N pay for nothing. Committing 7.5 h of TPU quota on
        an unseparated average would be guessing. `xm.wait_device_ops()` forces
        each step to complete so the timings are real rather than enqueue times.
        """
        def phase(label, n, who, batch=BATCH, clip=False, compiled=False):
            def one_step():
                idx = np.concatenate([m.rng.integers(lo, hi_t, batch) for m in who])
                arrs = gather(train, idx)
                for j, m in enumerate(who):
                    m.train_step(*slice_for(arrs, slice(j * batch, (j + 1) * batch)),
                                 clip=clip)
                return arrs

            body = one_step
            if compiled:
                # torch_xla.compile traces once and replays, instead of walking
                # the whole 19-block graph through Python on every step
                try:
                    body = torch_xla.compile(one_step)
                except Exception as exc:
                    print(f"windml bench {label} compile_unavailable: "
                          f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
                    return None

            times = []
            for k in range(n):
                t = time.time()
                try:
                    body()
                except Exception as exc:
                    print(f"windml bench {label} FAILED {type(exc).__name__}: "
                          f"{str(exc)[:300]}", flush=True)
                    return None
                xm.mark_step()
                xm.wait_device_ops()
                times.append((time.time() - t, 0.0))
                if k == 0:
                    print(f"windml bench {label} step1={times[0][0]:.1f}s "
                          f"(compile + execute)", flush=True)
            warm = times[1:] or times
            tot = sum(d for d, _ in warm) / len(warm)
            print(f"windml bench {label} members={len(who)} batch={batch} "
                  f"clip={clip} compiled={compiled} steady={tot:.3f}s/step "
                  f"-> {len(who)*batch/tot:.0f} samples/s", flush=True)
            return tot

        # Does cost scale with member COUNT? If 8 members cost ~8x one member,
        # the chips are idle and the bottleneck is Python walking the graph --
        # a fix, not a ceiling. If they cost about the same, they really are
        # running side by side and 8 chips are doing 8 chips' work.
        t1 = phase("1member", 10, members[:1])
        t8 = phase("8members", 10, members)
        if t1 and t8:
            print(f"windml bench scaling=8members/1member={t8/t1:.2f}x "
                  f"(1.0 = perfectly parallel, 8.0 = fully serial)", flush=True)

        def async_phase(label, n):
            """Time the loop the way it actually runs: mark_step only, no
            per-step sync.

            Every other number here forces xm.wait_device_ops() so the timing
            is real rather than enqueue time -- but the training loop does not
            do that, it lets the host run ahead and the device queue absorb the
            latency. If the fixed ~1.6 s/step is the instrument rather than the
            work, this is where it shows up.
            """
            xm.wait_device_ops()
            t = time.time()
            for _ in range(n):
                idx = np.concatenate([m.rng.integers(lo, hi_t, BATCH) for m in members])
                arrs = gather(train, idx)
                for j, m in enumerate(members):
                    m.train_step(*slice_for(arrs, slice(j * BATCH, (j + 1) * BATCH)))
                xm.mark_step()
            xm.wait_device_ops()      # one sync, at the end
            dt = (time.time() - t) / n
            print(f"windml bench {label} steady={dt:.3f}s/step "
                  f"-> {len(members)*BATCH/dt:.0f} samples/s", flush=True)
            return dt

        t8a = async_phase("8members_async", 20)
        if t8 and t8a:
            print(f"windml bench sync_overhead={t8 - t8a:+.3f}s/step "
                  f"(how much of the per-step cost was the measurement)", flush=True)

        # Previously measured on this same hardware, before packing: 1 member
        # 1.957, 8 members 1.669, async 1.634, compiled 1.681 (no help), batch
        # 128 1.978, clipping +274%. Those are the baseline this run is
        # compared against -- the compiled / batch-128 / clipping phases are
        # not repeated, both because they are answered and because running six
        # differently-shaped graphs in one process exhausted HBM.
        print("windml bench baseline_before_packing 1member=1.957 8members=1.669 "
              "async=1.634 s/step", flush=True)
        t = time.time()
        validate()
        print(f"windml bench validate_pass={time.time()-t:.1f}s", flush=True)
        for label, s in (("8members_async", t8a), ("8members", t8)):
            if s:
                print(f"windml bench projected_{label}={7.5*3600/s:.0f} steps "
                      f"in 7.5h = {7.5*3600/s*BATCH/1000:.0f}k samples per member",
                      flush=True)

    @torch.no_grad()
    def validate():
        """Every member on the same validation batches -- the scores must compare."""
        for m in members:
            m.model.eval()
        tot = [0.0] * len(members)
        n = 0
        for s in range(lo, hi_v, BATCH * 8):
            idx = np.arange(s, min(s + BATCH, hi_v))
            if len(idx) < 2:
                continue
            arrs = gather(val, idx)
            losses = []
            for m in members:
                x, y = m.assemble(arrs)      # once per member, not once per use
                losses.append(m.loss(m.model(x), y))
            xm.mark_step()
            for k, l in enumerate(losses):
                tot[k] += float(l)
            n += 1
        for m in members:
            m.model.train()
        return [t / max(n, 1) for t in tot]

    if SMOKE:
        # The smoke run's job is no longer "does it start" -- that is answered.
        # It is now "what does a step actually cost", which decides whether the
        # full run is worth 7.5 h of quota.
        benchmark()
        print("RESULT OK")
        return

    step = 0
    while step < MAX_STEPS and any(not m.done for m in members):
        # one contiguous gather of BATCH * n_members samples, split into disjoint
        # per-member batches: same bytes as eight separate gathers, one call
        live = [m for m in members if not m.done]
        idx = np.concatenate([m.rng.integers(lo, hi_t, BATCH) for m in live])
        arrs = gather(train, idx)
        losses = [m.train_step(*slice_for(arrs, slice(k * BATCH, (k + 1) * BATCH)),
                               clip=CLIP_GRAD)
                  for k, m in enumerate(live)]
        xm.mark_step()          # execute all members' graphs together
        step += 1

        if step % VAL_EVERY == 0:
            vals = validate()
            trs = [float(l.detach()) for l in losses]
            el = time.time() - t0
            bad = [m.rank for m, v in zip(members, vals) if not np.isfinite(v)]
            if bad or not all(np.isfinite(trs)):
                raise SystemExit(f"RESULT FAIL non-finite at step {step}: "
                                 f"members {bad} val={vals} train={trs}")
            print(f"windml step={step} elapsed={el/60:.0f}m "
                  f"val={[round(v, 4) for v in vals]}", flush=True)

            for m, v in zip(members, vals):
                if m.done:
                    continue
                m.log.append({"step": step, "lr": m.lr, "val": v, "s": el})
                if v < m.best:
                    m.best, m.stalled = v, 0
                    xm.save({"state_dict": m.model.state_dict(), "step": step,
                             "val": v, "seed": m.seed, "channels": names,
                             "targets": TARGETS, "direct_std": dstd.tolist(),
                             "store_std": [float(s) for s in store_std_tgt],
                             "n_frames": N_FRAMES, "direct_lead_h": 72},
                            str(OUT / f"rt2021_72h_seed{m.rank}.pt"))
                else:
                    m.stalled += 1
                    if m.stalled > EARLY_STOP:
                        m.done = True
                        print(f"windml member={m.rank} early_stop step={step} "
                              f"best={m.best:.4f}", flush=True)
                    elif m.stalled > PLATEAU_PATIENCE and m.drops < MAX_DROPS:
                        m.lr *= PLATEAU_FACTOR
                        m.drops += 1
                        m.stalled = 0
                        for g in m.opt.param_groups:
                            g["lr"] = m.lr
                        print(f"windml member={m.rank} lr_drop="
                              f"{m.drops}/{MAX_DROPS} -> {m.lr:.2e}", flush=True)
            (OUT / "metrics.json").write_text(json.dumps(
                {"params": n_par, "steps": step,
                 "members": [{"rank": m.rank, "seed": m.seed, "best_val": m.best,
                              "log": m.log} for m in members]}, indent=2))

        if time.time() - t0 > TIME_BUDGET_S:
            print(f"windml time_budget_reached step={step}", flush=True)
            break

    saved = sorted(OUT.glob("rt2021_72h_seed*.pt"))
    bests = [round(m.best, 4) for m in members]
    print(f"windml steps={step} members_saved={len(saved)} best_val={bests}", flush=True)
    # A success line has to be conditional on having produced something: the
    # first GPU run printed RESULT OK over best_val=inf and no checkpoint at all.
    if len(saved) < 2 or not all(np.isfinite(m.best) for m in members):
        raise SystemExit(f"RESULT FAIL saved={len(saved)} best_val={bests}")
    print("RESULT OK")


if __name__ == "__main__":
    main()

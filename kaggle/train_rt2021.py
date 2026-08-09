"""Rasp & Thuerey (2021) direct 72h model. Runs as a Kaggle GPU kernel.

The fidelity gate for the whole reproduction. Their ERA5-only model scores
z500 RMSE **314** at 3 days on 2017-2018; their published **268** is the
CMIP6-pretrained number. If this lands near 314 the copy is faithful and
pretraining is worth the quota. If it does not, pretraining would just be an
expensive way to paper over a bug.

Self-contained on purpose: a Kaggle kernel cannot import `windml`, so the
architecture and training recipe are duplicated here. Both are pinned by tests
in the repo (tests/test_rt_resnet.py asserts the 6.36M parameter count against
the paper's ~6.3M), and any change must be made in both places.

Reads /kaggle/input/windml-prep-era5 -- the prep kernel's output, mounted via
`kernel_sources`, so the 9 GB of arrays never move.

Everything below is from arXiv 2008.08626v2 section 2.2 cross-checked against
the author's src/networks.py. Two details are easy to get wrong and both are
deliberate: LeakyReLU comes BEFORE BatchNorm, and the weight decay is true L2
on convolution kernels only (Keras `kernel_regularizer`), which plain Adam
reproduces and AdamW does not.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy as np


def _ensure_torch_supports_this_gpu() -> None:
    """Install a torch with kernels for this GPU, if the preinstalled one lacks them.

    Kaggle hands out a Tesla P100, which is Pascal (sm_60), and recent torch
    wheels dropped Pascal. The failure is not a clean "unsupported device" --
    it is `CUDA error: no kernel image is available for execution on the
    device` on the first real op, several lines away from the cause. This is
    the same trap RUNBOOK.md gates against for the GTX 1080 (sm_61); the lesson
    just had not been applied here.

    The probe runs in a SUBPROCESS so that, if a reinstall is needed, the main
    process still imports torch fresh afterwards.
    """
    probe = ("import torch;"
             "cap=torch.cuda.get_device_capability(0);"
             "a='sm_%d%d'%cap;"
             "print(a, a in torch.cuda.get_arch_list(), torch.__version__)")

    def check() -> tuple[bool, str]:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        out = r.stdout.strip() or r.stderr.strip()[:140]
        return ("True" in r.stdout), out

    ok, out = check()
    print(f"windml gpu_arch_probe={out}", flush=True)
    if ok:
        return

    # cu121 does NOT carry Pascal either -- tried it, same
    # cudaErrorNoKernelImageForDevice. cu118 is the last build line that ships
    # sm_60/sm_61 kernels. Kaggle's kernel metadata exposes only `enable_gpu`
    # with no way to request a T4 (sm_75), so matching the wheel to the P100 is
    # the only lever available from the API.
    print("windml installing torch cu118 (last line with Pascal kernels) ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--index-url", "https://download.pytorch.org/whl/cu118",
                    "torch==2.4.1+cu118"], check=True)

    # Re-probe. Installing is not the same as fixing, and the failure it
    # prevents surfaces ~50 lines later as an unrelated-looking CUDA error.
    ok, out = check()
    print(f"windml gpu_arch_reprobe={out}", flush=True)
    if not ok:
        raise SystemExit(
            "RESULT FAIL no torch build available with kernels for this GPU "
            f"({out}). Switch the notebook to a T4 in the Kaggle UI, which is "
            "sm_75 and works with any current torch.")


_ensure_torch_supports_this_gpu()

import torch
from torch import nn

OUT = pathlib.Path("/kaggle/working")


def find_input() -> pathlib.Path:
    """Locate the prep kernel's output under /kaggle/input.

    Kaggle does not mount a kernel output at the slug you request it by --
    guessing /kaggle/input/windml-prep-era5 failed with FileNotFoundError on
    meta.json. Rather than hardcode another guess, find the directory that
    actually holds the data and say what was there if it is missing.
    """
    root = pathlib.Path("/kaggle/input")
    hits = sorted(root.glob("**/meta.json"))
    if hits:
        print(f"windml input_dir={hits[0].parent}", flush=True)
        return hits[0].parent
    listing = [str(p) for p in sorted(root.glob("*"))] or ["<empty>"]
    raise SystemExit(
        "RESULT FAIL no meta.json under /kaggle/input; mounted: " + ", ".join(listing))


IN = find_input()

TRAIN, VAL, TEST = (1979, 2015), (2016, 2016), (2017, 2018)
TARGETS = ["z500", "t850", "t2m"]
N_FRAMES = 3          # t, t-6h, t-12h
DIRECT_STEPS = 12     # 72 h at 6-hourly
LEAKY = 0.3

BATCH = 32
LR = 5e-5
WD = 1e-5
PLATEAU_FACTOR, PLATEAU_PATIENCE, MAX_DROPS, EARLY_STOP = 0.2, 2, 2, 5
VAL_EVERY = 2000
# Kaggle cuts GPU sessions at 12 h; stop with room to write the checkpoint and
# the metrics rather than being killed mid-save.
TIME_BUDGET_S = 10.5 * 3600


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
        self.stem = ConvBlock(cin, width, 7, p)      # kernel 7 widens the view
        self.blocks = nn.ModuleList(ResBlock(width, p) for _ in range(blocks))
        self.head = CircularConv2d(width, cout, 3)   # no activation on the head

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)

    def conv_kernels(self):
        return [m.conv.weight for m in self.modules() if isinstance(m, CircularConv2d)]


def load_years(lo, hi):
    """Concatenate the per-year arrays without doubling peak memory.

    37 train years is ~8.4 GB in fp16, and np.concatenate holds the input list
    AND the result at once -- ~17 GB, which risks OOM on a Kaggle kernel. Read
    the headers first, allocate once, then fill in place.
    """
    years = list(range(lo, hi + 1))
    shapes = [np.load(IN / f"era5_rt2021_{y}.npy", mmap_mode="r").shape for y in years]
    total = sum(s[0] for s in shapes)
    out = np.empty((total, *shapes[0][1:]), dtype=np.float16)
    at = 0
    for y, s in zip(years, shapes):
        out[at:at + s[0]] = np.load(IN / f"era5_rt2021_{y}.npy")
        at += s[0]
    return out


def latitude_field(n_lat, n_lon):
    """sin(latitude) as a plane -- the third of the paper's three constants.

    Given as sin(lat) rather than degrees: same information, bounded, and it
    matches how the repo's own static channels are built.
    """
    lat = np.linspace(-87.1875, 87.1875, n_lat, dtype=np.float32)
    return np.broadcast_to(np.sin(np.deg2rad(lat))[:, None],
                           (n_lat, n_lon)).astype(np.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"windml device={dev} name="
          f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}", flush=True)

    meta = json.loads((IN / "meta.json").read_text())
    stats = json.loads((IN / "stats.json").read_text())
    names = meta["channels"]
    if not stats.get("stored_normalized"):
        raise SystemExit(
            "RESULT FAIL this input stores physical units in float16, which "
            "overflows for z50/z250 and yields inf. Re-run the prep kernel.")

    # The arrays arrive as (physical - store_mean)/store_std. The network wants
    # (physical - mean)/std against the exact train statistics, and composing
    # the two affines is one multiply-add rather than a round trip through
    # physical units -- which is also the only form that stays float16-safe.
    sm = np.asarray(stats["store_mean"], np.float32)[None, :, None, None]
    ss = np.asarray(stats["store_std"], np.float32)[None, :, None, None]
    mean = np.asarray(stats["mean"], np.float32)[None, :, None, None]
    std = np.asarray(stats["std"], np.float32)[None, :, None, None]
    re_a = ss / std
    re_b = (sm - mean) / std
    tgt = [names.index(t) for t in TARGETS]
    print(f"windml channels={len(names)} targets={TARGETS}->{tgt} "
          f"stored_normalized=True", flush=True)

    train = load_years(*TRAIN)
    val = load_years(*VAL)
    n_lat, n_lon = train.shape[2], train.shape[3]
    print(f"windml train={train.shape} val={val.shape}", flush=True)

    lat_static = torch.from_numpy(latitude_field(n_lat, n_lon)).to(dev)
    # The paper's other two constants are land-sea mask and orography, which
    # the prep kernel does not ship. Both are derived from the data: 925 hPa
    # geopotential is depressed over high terrain, so its time mean stands in
    # for orography and a threshold on it for the mask. Cruder than ERA5's own
    # fields, but they are constants the network learns around, and keeping the
    # count at 3 is what holds the input to the paper's 117.
    zsurf = torch.from_numpy(
        (train[::200, names.index("z925")].astype(np.float32)).mean(0)).to(dev)
    zsurf = (zsurf - zsurf.mean()) / (zsurf.std() + 1e-8)
    lsm = (zsurf > 0.15).float()
    const = torch.stack([lsm, zsurf, lat_static])[None]      # (1, 3, H, W)

    n_in = len(names) * N_FRAMES + 3
    model = WeatherResNetRT(cin=n_in, cout=len(TARGETS)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"windml in_channels={n_in} params={n_par/1e6:.3f}M "
          f"(paper ~6.3M)", flush=True)
    assert n_in == 117, f"expected the paper's 117 inputs, got {n_in}"

    kern = {id(p) for p in model.conv_kernels()}
    opt = torch.optim.Adam(
        [{"params": [p for p in model.parameters() if id(p) in kern], "weight_decay": WD},
         {"params": [p for p in model.parameters() if id(p) not in kern], "weight_decay": 0.0}],
        lr=LR)

    re_a_t = torch.from_numpy(re_a).to(dev)
    re_b_t = torch.from_numpy(re_b).to(dev)
    store_std_t = torch.from_numpy(ss[:, tgt]).to(dev)
    latw = np.cos(np.deg2rad(np.linspace(-87.1875, 87.1875, n_lat)))
    latw = torch.from_numpy((latw / latw.mean()).astype(np.float32)).to(dev)[None, None, :, None]

    def direct_std(arr):
        rng = np.random.default_rng(0)
        hi = arr.shape[0] - DIRECT_STEPS - 1
        i = rng.choice(hi, size=min(1500, hi), replace=False)
        d = arr[i + DIRECT_STEPS][:, tgt].astype(np.float32) - arr[i][:, tgt].astype(np.float32)
        return torch.from_numpy(d.std(axis=(0, 2, 3)).astype(np.float32)).to(dev)[None, :, None, None]

    # in STORED units; multiplying by store_std recovers the physical spread,
    # which is the number worth eyeballing (z500 should be ~1030 m2/s2)
    dstd = direct_std(train)
    dstd_phys = (dstd * store_std_t).flatten().tolist()
    print(f"windml direct_std_stored={[round(v, 4) for v in dstd.flatten().tolist()]} "
          f"direct_std_physical={[round(v, 2) for v in dstd_phys]}", flush=True)

    def batch(arr, idx):
        """Assemble (x, y): 3 normalized frames + 3 constants -> scaled residual."""
        xs = []
        for k in range(N_FRAMES):
            f = torch.from_numpy(arr[idx - k].astype(np.float32)).to(dev)
            xs.append(re_a_t * f + re_b_t)
        x = torch.cat(xs + [const.expand(len(idx), -1, -1, -1)], 1)
        fut = torch.from_numpy(arr[idx + DIRECT_STEPS][:, tgt].astype(np.float32)).to(dev)
        now = torch.from_numpy(arr[idx][:, tgt].astype(np.float32)).to(dev)
        return x, (fut - now) / dstd

    def loss_fn(p, y):
        return (latw * (p - y) ** 2).mean()

    lo_t, hi_t = N_FRAMES - 1, train.shape[0] - DIRECT_STEPS - 1
    lo_v, hi_v = N_FRAMES - 1, val.shape[0] - DIRECT_STEPS - 1
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for s in range(lo_v, hi_v, BATCH * 8):
            idx = np.arange(s, min(s + BATCH, hi_v))
            if len(idx) < 2:
                continue
            x, y = batch(val, idx)
            tot += float(loss_fn(model(x), y)); n += 1
        model.train()
        return tot / max(n, 1)

    best, stalled, drops, lr = float("inf"), 0, 0, LR
    t0, step = time.time(), 0
    log = []
    while True:
        idx = rng.integers(lo_t, hi_t, BATCH)
        x, y = batch(train, idx)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % VAL_EVERY == 0:
            v = validate()
            # Bail on the FIRST non-finite validation. The previous run spent
            # 136 GPU-minutes and 24,000 steps demonstrating that NaN stays
            # NaN; the cause was inf in the inputs, and no amount of further
            # training was going to change it.
            if not np.isfinite(v) or not np.isfinite(float(loss)):
                raise SystemExit(
                    f"RESULT FAIL non-finite at step {step}: "
                    f"train={float(loss)} val={v}. Inputs or targets contain "
                    f"inf/nan -- check the prep kernel, not the recipe.")
            el = time.time() - t0
            log.append({"step": step, "lr": lr, "train": float(loss), "val": v, "s": el})
            print(f"windml step={step} lr={lr:.2e} train={float(loss):.4f} "
                  f"val={v:.4f} elapsed={el/60:.0f}m", flush=True)
            if v < best:
                best, stalled = v, 0
                torch.save({"state_dict": model.state_dict(), "step": step,
                            "val": v, "channels": names, "targets": TARGETS,
                            "direct_std": dstd.cpu().numpy().tolist(),
                            "direct_std_physical": dstd_phys,
                            "store_mean": sm[:, tgt].ravel().tolist(),
                            "store_std": ss[:, tgt].ravel().tolist(),
                            "n_frames": N_FRAMES, "direct_lead_h": 72},
                           OUT / "rt2021_72h_best.pt")
            else:
                stalled += 1
                if stalled > EARLY_STOP:
                    print(f"windml early_stop step={step} best={best:.4f}")
                    break
                if stalled > PLATEAU_PATIENCE and drops < MAX_DROPS:
                    lr *= PLATEAU_FACTOR; drops += 1; stalled = 0
                    for g in opt.param_groups:
                        g["lr"] = lr
                    print(f"windml lr_drop={drops}/{MAX_DROPS} -> {lr:.2e}", flush=True)
            (OUT / "metrics.json").write_text(json.dumps(
                {"log": log, "best_val": best, "params": n_par}, indent=2))

        if time.time() - t0 > TIME_BUDGET_S:
            print(f"windml time_budget_reached step={step}")
            break

    ckpt = OUT / "rt2021_72h_best.pt"
    print(f"windml best_val={best:.4f} steps={step}")
    # A success line has to be conditional on having produced something. The
    # last run printed RESULT OK over best_val=inf and no checkpoint at all.
    if not np.isfinite(best) or not ckpt.exists():
        raise SystemExit(f"RESULT FAIL best_val={best} checkpoint={ckpt.exists()}")
    print("RESULT OK")


if __name__ == "__main__":
    main()

"""Score the RT2021 direct-72h model on 2017-2018. The gate, in physical units.

The training kernel reports a validation loss on scaled residuals, which is not
comparable to anything in the literature. This produces the number that is:
latitude-weighted RMSE for z500, t850 and t2m at 72 h on the paper's own test
years, against

    Direct (ERA5 only)        z500 314 / t850 1.79 / t2m 1.53
    Direct (CMIP6-pretrained) z500 268 / t850 1.65 / t2m 1.42

**314 is the target here, not 268** -- 268 requires pretraining on ~150 years of
CMIP6 and this model has seen only ERA5.

Runs on Kaggle for the same reason prep does: the 2017-2018 inputs are part of
the 9 GB array set, which cannot come to the dev container. Mounts the prep
kernel's output for the data and the training kernel's output for the weights.

Units. The arrays are stored as (physical - store_mean)/store_std, and the model
predicts a residual scaled by direct_std, so

    pred_stored  = now_stored + out * direct_std
    error_phys   = (pred_stored - truth_stored) * store_std

and the RMSE in physical units is the latitude-weighted RMS of that. No round
trip through physical fields is needed, and store_mean cancels in the
difference.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np


def _ensure_torch_supports_this_gpu() -> None:
    """Install a torch with kernels for this GPU, if the preinstalled one lacks them.

    Copied from train_rt2021.py, which is the point: the first run of this
    kernel died with `CUDA error: no kernel image is available for execution on
    the device` because the gate existed in the training kernel and had simply
    not been applied here. Kaggle hands out a Tesla P100 (Pascal, sm_60) and
    current torch wheels dropped Pascal; cu118 is the last line that ships it.

    The probe runs in a SUBPROCESS so the main process imports torch fresh
    after any reinstall.
    """
    probe = ("import torch;"
             "cap=torch.cuda.get_device_capability(0);"
             "a='sm_%d%d'%cap;"
             "print(a, a in torch.cuda.get_arch_list(), torch.__version__)")

    def check():
        r = subprocess.run([sys.executable, "-c", probe],
                           capture_output=True, text=True)
        return ("True" in r.stdout), (r.stdout.strip() or r.stderr.strip()[:140])

    ok, out = check()
    print(f"windml gpu_arch_probe={out}", flush=True)
    if ok:
        return
    print("windml installing torch cu118 (last line with Pascal kernels) ...",
          flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--index-url", "https://download.pytorch.org/whl/cu118",
                    "torch==2.4.1+cu118"], check=True)
    ok, out = check()
    print(f"windml gpu_arch_reprobe={out}", flush=True)
    if not ok:
        raise SystemExit(f"RESULT FAIL no torch build with kernels for this GPU ({out})")


_ensure_torch_supports_this_gpu()

import torch  # noqa: E402
from torch import nn  # noqa: E402

OUT = pathlib.Path("/kaggle/working")
TEST = (2017, 2018)
TRAIN = (1979, 2015)
TARGETS = ["z500", "t850", "t2m"]
N_FRAMES = 3
LEAKY = 0.3
BATCH = 32

# The paper's own figures at each lead, for the comparison this kernel exists to
# make. Table 1 gives 3 d and 5 d; the shorter leads are not tabulated there, so
# they are scored and reported without a published counterpart rather than
# against a number invented for the occasion.
PAPER = {
    72: {"era5_only": {"z500": 314.0, "t850": 1.79, "t2m": 1.53},
         "pretrained": {"z500": 268.0, "t850": 1.65, "t2m": 1.42}},
    120: {"era5_only": {"z500": 561.0, "t850": 2.82, "t2m": 2.32},
          "pretrained": {"z500": 523.0, "t850": 2.52, "t2m": 2.03}},
}


class CircularConv2d(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(cin, cout, k)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = nn.functional.pad(x, (0, 0, self.pad, self.pad))
        return self.conv(x)


class ConvBlock(nn.Module):
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


class YearStack:
    def __init__(self, root, lo, hi, eager=False):
        self.parts = [np.load(root / f"era5_rt2021_{y}.npy",
                              mmap_mode=None if eager else "r")
                      for y in range(lo, hi + 1)]
        self.offsets = np.cumsum([0] + [p.shape[0] for p in self.parts])
        self.shape = (int(self.offsets[-1]), *self.parts[0].shape[1:])

    def take(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), *self.shape[1:]), dtype=np.float32)
        part_of = np.searchsorted(self.offsets, idx, side="right") - 1
        for p in np.unique(part_of):
            sel = part_of == p
            out[sel] = self.parts[p][idx[sel] - self.offsets[p]].astype(np.float32)
        return out


def latitude_field(n_lat, n_lon):
    lat = np.linspace(-87.1875, 87.1875, n_lat, dtype=np.float32)
    return np.broadcast_to(np.sin(np.deg2rad(lat))[:, None],
                           (n_lat, n_lon)).astype(np.float32)


def find_all(pattern):
    hits = sorted(pathlib.Path("/kaggle/input").glob(pattern))
    if not hits:
        listing = [str(p) for p in sorted(pathlib.Path("/kaggle/input").glob("*"))]
        raise SystemExit(f"RESULT FAIL no {pattern} under /kaggle/input; "
                         f"mounted: {', '.join(listing) or '<empty>'}")
    return hits


def find(pattern):
    return find_all(pattern)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = find("**/meta.json").parent
    # One checkpoint from the GPU run, or eight from the TPU ensemble -- the
    # ensemble IS the deliverable there, so scoring one member and calling it
    # the result would miss the point of the run.
    all_paths = find_all("**/rt2021_*h_*.pt")
    print(f"windml data={src} checkpoints={len(all_paths)} device={dev}", flush=True)
    for p in all_paths:
        print(f"windml   ckpt={p.name}", flush=True)

    meta = json.loads((src / "meta.json").read_text())
    stats = json.loads((src / "stats.json").read_text())
    names = meta["channels"]
    tgt = [names.index(t) for t in TARGETS]

    sm = np.asarray(stats["store_mean"], np.float32)[None, :, None, None]
    ss = np.asarray(stats["store_std"], np.float32)[None, :, None, None]
    mean = np.asarray(stats["mean"], np.float32)[None, :, None, None]
    std = np.asarray(stats["std"], np.float32)[None, :, None, None]
    re_a = np.tile(ss / std, (1, N_FRAMES, 1, 1))
    re_b = np.tile((sm - mean) / std, (1, N_FRAMES, 1, 1))
    store_std_tgt = ss[0, tgt, 0, 0]                       # (3,)

    # Group by the lead each model was TRAINED for. Averaging a 72 h and a 120 h
    # forecast would produce a number that looks like an ensemble and means
    # nothing, and the lead also sets which truth frame is the target -- so it
    # is read from the checkpoint rather than assumed.
    groups: dict[int, list] = {}
    for p in all_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck.get("targets") != TARGETS:
            raise SystemExit(f"RESULT FAIL {p.name} targets {ck.get('targets')} "
                             f"!= {TARGETS}; the scoring would be mislabelled")
        lead = ck.get("direct_lead_h")
        if not lead or lead % 6:
            raise SystemExit(f"RESULT FAIL {p.name} has direct_lead_h={lead!r}; "
                             f"cannot tell which lead it forecasts")
        groups.setdefault(int(lead), []).append((p, ck))
        print(f"windml {p.stem} lead={lead}h step={ck.get('step')} "
              f"val={ck.get('val')} seed={ck.get('seed')}", flush=True)
    print(f"windml lead_groups="
          f"{ {k: len(v) for k, v in sorted(groups.items())} }", flush=True)

    test = YearStack(src, *TEST)
    n_lat, n_lon = test.shape[2], test.shape[3]

    # The three constants have to match training EXACTLY or the model is fed a
    # different input than it was fit on. The training kernel derived them from
    # the concatenated train array at stride 200, so that is reproduced here
    # rather than approximated. (Storing them in the checkpoint would be less
    # fragile; that is a change for the next training run, not this scoring.)
    train = YearStack(src, *TRAIN)
    probe = train.take(np.arange(0, train.shape[0], 200))
    zsurf = torch.from_numpy(probe[:, names.index("z925")].mean(0))
    zsurf = (zsurf - zsurf.mean()) / (zsurf.std() + 1e-8)
    const = torch.stack([(zsurf > 0.15).float(), zsurf,
                         torch.from_numpy(latitude_field(n_lat, n_lon))])[None].to(dev)
    del probe, train

    # WeatherBench weighting: cos(latitude), normalized to mean 1
    w = np.cos(np.deg2rad(np.linspace(-87.1875, 87.1875, n_lat))).astype(np.float32)
    w = w / w.mean()
    wt = torch.from_numpy(w).to(dev)[None, None, :, None]

    re_a_t = torch.from_numpy(re_a).to(dev)
    re_b_t = torch.from_numpy(re_b).to(dev)
    sst = torch.from_numpy(store_std_tgt.copy()).to(dev)
    lo = N_FRAMES - 1

    def score_lead(lead, entries):
        """Members and their plain-mean ensemble, for one lead."""
        steps = lead // 6
        models, dstd_t = [], []
        for _p, ck in entries:
            mdl = WeatherResNetRT(cin=len(names) * N_FRAMES + 3,
                                  cout=len(TARGETS)).to(dev)
            mdl.load_state_dict(ck["state_dict"])
            mdl.eval()
            models.append(mdl)
            dstd_t.append(torch.from_numpy(np.asarray(
                ck["direct_std"], np.float32).reshape(1, len(TARGETS), 1, 1)).to(dev))

        inits = np.arange(lo, test.shape[0] - steps)
        print(f"windml lead={lead}h members={len(models)} inits={len(inits)}",
              flush=True)

        sq = [torch.zeros(len(TARGETS), device=dev) for _ in models]
        sq_ens = torch.zeros(len(TARGETS), device=dev)
        n = 0
        with torch.no_grad():
            for s in range(0, len(inits), BATCH):
                idx = inits[s:s + BATCH]
                now = test.take(idx)
                frames = np.concatenate(
                    [now] + [test.take(idx - k) for k in range(1, N_FRAMES)], axis=1)
                d = torch.from_numpy(frames).to(dev)
                x = torch.cat([d * re_a_t + re_b_t,
                               const.expand(len(idx), -1, -1, -1)], 1)
                now_t = torch.from_numpy(now[:, tgt]).to(dev)
                truth = torch.from_numpy(test.take(idx + steps)[:, tgt]).to(dev)

                acc = None
                for k, mdl in enumerate(models):
                    pred = now_t + mdl(x) * dstd_t[k]      # both in stored units
                    sq[k] += (wt * (pred - truth) ** 2).mean(dim=(2, 3)).sum(dim=0)
                    acc = pred if acc is None else acc + pred
                # plain mean of member forecasts: no fitted weights, which is
                # what made the earlier avg5 result in this repo credible
                sq_ens += (wt * (acc / len(models) - truth) ** 2
                           ).mean(dim=(2, 3)).sum(dim=0)
                n += len(idx)
                if s % (BATCH * 40) == 0:
                    print(f"windml   lead={lead}h scored={n}/{len(inits)}", flush=True)

        # stored -> physical: store_mean cancels in a difference, only the scale
        def to_rmse(a):
            r = (torch.sqrt(a / n) * sst).cpu().numpy()
            return {t: float(v) for t, v in zip(TARGETS, r)}

        members = [to_rmse(a) for a in sq]
        ensemble = to_rmse(sq_ens)
        headline = ensemble if len(models) > 1 else members[0]
        if not all(np.isfinite(list(headline.values()))):
            raise SystemExit(f"RESULT FAIL non-finite RMSE at {lead}h: {headline}")

        ref = PAPER.get(lead)
        print("windml " + "=" * 66, flush=True)
        for t in TARGETS:
            ours = headline[t]
            if ref:
                era, pre = ref["era5_only"][t], ref["pretrained"][t]
                print(f"windml {t:>5s} @{lead}h  ours={ours:8.2f}  "
                      f"paper_era5_only={era:7.2f} ({100*(ours-era)/era:+.1f}%)  "
                      f"paper_pretrained={pre:7.2f}", flush=True)
            else:
                # no published figure at this lead -- say so rather than
                # comparing against a number invented for the occasion
                print(f"windml {t:>5s} @{lead}h  ours={ours:8.2f}  "
                      f"(paper does not tabulate this lead)", flush=True)
        if len(models) > 1:
            best = min(m["z500"] for m in members)
            print(f"windml ensemble_gain @{lead}h z500 best_member={best:.2f} -> "
                  f"ensemble={ensemble['z500']:.2f} "
                  f"({100*(ensemble['z500']-best)/best:+.1f}%)", flush=True)
        print("windml " + "=" * 66, flush=True)

        return {"lead_h": lead, "rmse": headline, "members": members,
                "ensemble": ensemble if len(models) > 1 else None,
                "n_members": len(models), "n_inits": int(n), "paper": ref,
                "ckpts": [{"file": p.name, "step": c.get("step"),
                           "val": c.get("val"), "seed": c.get("seed")}
                          for p, c in entries]}

    results = [score_lead(lead, entries) for lead, entries in sorted(groups.items())]
    (OUT / "rt2021_scores.json").write_text(json.dumps({
        "by_lead": results, "test_years": TEST,
        "gate": "z500 near 314 at 72h (ERA5-only); 268 needs CMIP6 pretraining",
    }, indent=2))
    print("RESULT OK")


if __name__ == "__main__":
    main()

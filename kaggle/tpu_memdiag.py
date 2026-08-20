"""Why does the TPU ensemble OOM on real data but not synthetic? Diagnostic kernel.

Three full runs died in the same place -- the first host->device transfer --
with the allocator reporting almost nothing free:

    batch 128   allocating 120 MB, 21.4 MB free
    batch  64   allocating  64 MB, 37.2 MB free
    batch  32   allocating  32 MB,  5.0 MB free

Two hypotheses are already dead. Batch size is not it: halving the batch
recovered only the amount the input block itself shrank by. Member
co-location is not it either: a per-device readout showed all eight chips at
0.03 GB used of a 16.91 GB limit, so the members are correctly spread.

That readout is also the puzzle. `get_memory_info` claims ~16.9 GB free on the
very device whose allocator then says 5 MB. Something is consuming memory that
the device memory info does not describe.

**The untested variable is the data source.** Synthetic x 8 devices WORKS -- the
smoke run trained eight full 6.36M models and saved eight checkpoints. Real
data x 8 devices fails. The largest difference between those paths is host RAM:
YearStack(eager=True) pulls ~8.4 GB of ERA5 into the process, where
SyntheticStack holds ~190 MB. If libtpu stages transfers through host memory,
a nearly-full host would explain an allocation failure that device-side
accounting cannot see.

So this kernel walks the 2x2 and prints host memory alongside device memory at
every step:

    A  synthetic, 8 devices      (known good -- the control)
    B  real eager, 1 device      is it the data, or the data x 8 devices?
    C  real mmap,  8 devices     does dropping 8.4 GB of host RAM fix it?
    D  real eager, 8 devices     the known failure, run last so it cannot
                                 poison the cases before it

Each case is isolated in a try/except so one failure does not hide the rest --
the whole point is to see which combination breaks.
"""

import json
import pathlib
import traceback

import numpy as np
import torch
from torch import nn

import torch_xla
import torch_xla.core.xla_model as xm

N_FRAMES = 3
DIRECT_STEPS = 12
TARGETS = ["z500", "t850", "t2m"]
BATCH = 32
LEAKY = 0.3
TRAIN = (1979, 2015)


def host_mem() -> str:
    """MemAvailable is the number that matters -- MemFree ignores page cache."""
    try:
        info = {}
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k] = v.strip()
        return (f"avail={info.get('MemAvailable', '?')} "
                f"total={info.get('MemTotal', '?')}")
    except OSError:
        return "unavailable"


def dev_mem(devices, label: str) -> None:
    print(f"windml [{label}] host {host_mem()}", flush=True)
    for d in devices[:2] + ([devices[-1]] if len(devices) > 2 else []):
        try:
            mi = xm.get_memory_info(d)
            print(f"windml [{label}]   {d} used={mi.get('bytes_used', 0)/1e9:.3f}GB "
                  f"limit={mi.get('bytes_limit', 0)/1e9:.2f}GB", flush=True)
        except Exception as exc:
            print(f"windml [{label}]   {d} memory_info failed: "
                  f"{type(exc).__name__}", flush=True)


# ------------------------------------------------------- architecture (as shipped)


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
        self.drop = nn.Dropout(p)

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
    def __init__(self, root, lo, hi, eager=True):
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


class SyntheticStack:
    def __init__(self, n_t=600, n_c=38, h=32, w=64, seed=0):
        self.shape = (n_t, n_c, h, w)
        self._buf = np.random.default_rng(seed).normal(
            0, 1, (n_t, n_c, h, w)).astype(np.float32)

    def take(self, idx):
        return self._buf[np.asarray(idx)]


def find_input():
    hits = sorted(pathlib.Path("/kaggle/input").glob("**/meta.json"))
    if not hits:
        raise SystemExit("RESULT FAIL no meta.json under /kaggle/input")
    return hits[0].parent


def trial(label, stack, devices, n_steps=3):
    """One model per device, real training steps, memory printed throughout."""
    n_c = stack.shape[1]
    n_in = n_c * N_FRAMES + 3
    rng = np.random.default_rng(0)
    dev_mem(devices, f"{label}/pre-build")

    models, opts = [], []
    for d in devices:
        m = WeatherResNetRT(cin=n_in, cout=len(TARGETS)).to(d)
        models.append(m)
        opts.append(torch.optim.Adam(m.parameters(), lr=5e-5))
    xm.mark_step()
    xm.wait_device_ops()
    dev_mem(devices, f"{label}/post-build")

    const = torch.zeros(1, 3, stack.shape[2], stack.shape[3])
    consts = [const.to(d) for d in devices]
    lo, hi = N_FRAMES - 1, stack.shape[0] - DIRECT_STEPS - 1

    for step in range(n_steps):
        idx = np.concatenate([rng.integers(lo, hi, BATCH) for _ in devices])
        now = stack.take(idx)
        blk = np.concatenate(
            [now] + [stack.take(idx - k) for k in range(1, N_FRAMES)]
            + [stack.take(idx + DIRECT_STEPS)[:, :3], now[:, :3]], axis=1)
        for j, (d, m, o) in enumerate(zip(devices, models, opts)):
            part = np.ascontiguousarray(blk[j * BATCH:(j + 1) * BATCH])
            t = torch.from_numpy(part).to(d)          # <-- where every run died
            x = torch.cat([t[:, :n_c * N_FRAMES],
                           consts[j].expand(BATCH, -1, -1, -1)], 1)
            y = t[:, n_c * N_FRAMES:n_c * N_FRAMES + 3] - t[:, -3:]
            loss = ((m(x) - y) ** 2).mean()
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
            if step == 0 and j in (0, len(devices) - 1):
                dev_mem(devices, f"{label}/step0-member{j}")
        xm.mark_step()
        xm.wait_device_ops()
    dev_mem(devices, f"{label}/done")


def main():
    devices = xm.get_xla_supported_devices()
    print(f"windml torch_xla={torch_xla.__version__} devices={len(devices)}", flush=True)
    print(f"windml host_at_start {host_mem()}", flush=True)
    src = find_input()
    meta = json.loads((src / "meta.json").read_text())
    print(f"windml channels={len(meta['channels'])}", flush=True)

    def run(label, fn):
        print("-" * 70, flush=True)
        print(f"windml CASE {label}", flush=True)
        try:
            fn()
            print(f"windml CASE {label} RESULT=ok", flush=True)
        except Exception as exc:
            print(f"windml CASE {label} RESULT=fail {type(exc).__name__}: "
                  f"{str(exc)[:300]}", flush=True)
            traceback.print_exc()

    # A: the control -- known to work, so a failure here means something else changed
    run("A_synthetic_8dev", lambda: trial("A", SyntheticStack(), devices))

    # B: real data, ONE device. Isolates "the data" from "the data x 8 devices".
    def case_b():
        st = YearStack(src, *TRAIN, eager=True)
        print(f"windml B loaded eager {st.shape}, host {host_mem()}", flush=True)
        trial("B", st, devices[:1])
    run("B_real_eager_1dev", case_b)

    # C: real data, 8 devices, MMAP. The candidate fix -- same data, ~8.4 GB less
    # host RAM. The eager load existed to stop 8 PROCESSES duplicating the array,
    # and the design is single-process now, so this costs nothing.
    def case_c():
        st = YearStack(src, *TRAIN, eager=False)
        print(f"windml C mmapped {st.shape}, host {host_mem()}", flush=True)
        trial("C", st, devices)
    run("C_real_mmap_8dev", case_c)

    # D: the known failure, last so it cannot poison the cases above
    def case_d():
        st = YearStack(src, *TRAIN, eager=True)
        print(f"windml D loaded eager {st.shape}, host {host_mem()}", flush=True)
        trial("D", st, devices)
    run("D_real_eager_8dev", case_d)

    print("=" * 70, flush=True)
    print("RESULT OK")


if __name__ == "__main__":
    main()

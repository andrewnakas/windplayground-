#!/usr/bin/env python3
"""Drive Kaggle kernels from here: render, push, poll, fetch.

Kaggle gives ~30 GPU-h and ~20 TPU-h a week, which is the compute this project
has been missing. `kaggle kernels push` runs a script on that hardware and
`kernels output` brings the results back, so the whole RT2021 reproduction can
run remotely without anyone sitting in a notebook.

Auth: the new-style `KGAT_...` tokens authenticate through **KAGGLE_API_TOKEN**,
not the username/key pair in kaggle.json. Passing one as a Basic-auth password
fails with a bare "Authentication required" that looks like a bad key. Source
~/.config/windml/kaggle.env (mode 600, outside the repo) before running.

    python scripts/kaggle_run.py push prep_era5
    python scripts/kaggle_run.py status prep_era5
    python scripts/kaggle_run.py output prep_era5 --dest artifacts/kaggle
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO / "kaggle"
STAGE = REPO / "artifacts" / "kaggle" / "_stage"
USER = os.environ.get("KAGGLE_USERNAME", "andrewnakas")

# Each entry becomes one Kaggle kernel. `gpu`/`tpu` pick the accelerator;
# `datasets` mounts previously-produced kernel outputs so a training run can
# read the arrays a prep run wrote.
KERNELS: dict[str, dict] = {
    "prep_era5": {
        "title": "windml prep era5",
        "source": "prep_era5.py",
        "gpu": False,
        "internet": True,
        "datasets": [],
        "note": "streams WeatherBench-2 zarr -> fp16 arrays + analytic TISR",
    },
    "train_rt2021": {
        "title": "windml train rt2021",
        "source": "train_rt2021.py",
        "gpu": True,
        "internet": True,
        "datasets": [],
        # Mount the prep kernel's OUTPUT directly. It landed at
        # /kaggle/input/windml-prep-era5, so the 9 GB of arrays never move --
        # they do not fit in this container anyway (pulling them once took the
        # disk to 100%). An earlier version pointed at a windml-era5-rt2021
        # dataset that was never created.
        "kernels": [f"{USER}/windml-prep-era5"],
        "note": "RT2021 ResNet, direct 72h, the 314 fidelity gate",
    },
    # TPU runs one WHOLE model per core -- eight seeds, not one data-parallel
    # model (see the kernel docstring). The smoke variant needs no data at all,
    # so it can prove the 8-core plumbing before any quota goes on a real run.
    "eval_rt2021": {
        "title": "windml eval rt2021",
        "source": "eval_rt2021.py",
        "gpu": True,
        "internet": True,
        # both the arrays and the weights live as kernel outputs, so nothing
        # has to be uploaded or published as a dataset
        "kernels": [f"{USER}/windml-prep-era5", f"{USER}/windml-train-rt2021"],
        "note": "z500/t850/t2m RMSE @72h on 2017-2018 -- the number vs 314",
    },
    "tpu_probe": {
        "title": "windml tpu probe",
        "source": "tpu_probe.py",
        "tpu": True,
        "internet": True,
        "note": "what does a Kaggle TPU look like to torch_xla? env + 3 init paths",
    },
    "tpu_smoke": {
        "title": "windml tpu smoke",
        "source": "train_rt2021_tpu.py",
        "tpu": True,
        "internet": True,
        "prelude": 'import os; os.environ["WINDML_SMOKE"] = "1"\n',
        "note": "synthetic data, ~5 min: proves 8 cores + fwd/bwd + per-core save",
    },
    "train_rt2021_tpu": {
        "title": "windml train rt2021 tpu ensemble",
        "source": "train_rt2021_tpu.py",
        "tpu": True,
        "internet": True,
        "kernels": [f"{USER}/windml-prep-era5"],
        "note": "8 independent seeds, one per core -> the ensemble directly",
    },
}


def _kaggle_bin() -> str:
    """The CLI usually lives in the venv, not on PATH."""
    for cand in (REPO / ".venv" / "bin" / "kaggle",
                 pathlib.Path(sys.executable).parent / "kaggle"):
        if cand.exists():
            return str(cand)
    found = shutil.which("kaggle")
    if not found:
        sys.exit("kaggle CLI not found (pip install kaggle)")
    return found


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    if "KAGGLE_API_TOKEN" not in os.environ:
        # The new KGAT_ tokens authenticate through this variable only; as a
        # Basic-auth password they fail with a bare "Authentication required".
        sys.exit("KAGGLE_API_TOKEN is not set. "
                 "Run: set -a; . ~/.config/windml/kaggle.env; set +a")
    args = [_kaggle_bin(), *args[1:]] if args[0] == "kaggle" else args
    return subprocess.run(args, capture_output=True, text=True, check=check)


def slug(name: str) -> str:
    return f"{USER}/windml-{name.replace('_', '-')}"


def stage(name: str) -> pathlib.Path:
    """Write the kernel source and its metadata into a clean directory."""
    spec = KERNELS[name]
    src = KERNEL_DIR / spec["source"]
    if not src.exists():
        sys.exit(f"missing kernel source: {src}")

    out = STAGE / name
    out.mkdir(parents=True, exist_ok=True)
    # Kaggle has no way to set an environment variable on a kernel, so a source
    # shared by two kernels is configured by prepending a line to it.
    (out / spec["source"]).write_text(spec.get("prelude", "") + src.read_text())

    meta = {
        "id": slug(name),
        "title": spec["title"],
        "code_file": spec["source"],
        "language": "python",
        "kernel_type": "script",
        "is_private": False,
        "enable_gpu": bool(spec.get("gpu")),
        "enable_tpu": bool(spec.get("tpu")),
        "enable_internet": bool(spec.get("internet", True)),
        "dataset_sources": spec.get("datasets", []),
        "competition_sources": [],
        "kernel_sources": spec.get("kernels", []),
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def push(name: str) -> None:
    d = stage(name)
    r = _run(["kaggle", "kernels", "push", "-p", str(d)], check=False)
    print((r.stdout + r.stderr).strip())
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"windml pushed={slug(name)}")
    print(f"windml url=https://www.kaggle.com/code/{slug(name).replace('/', '/')}")


def status(name: str) -> str:
    r = _run(["kaggle", "kernels", "status", slug(name)], check=False)
    text = (r.stdout + r.stderr).strip()
    print(text)
    # the CLI reports KernelWorkerStatus.RUNNING / .ERROR / .COMPLETE, so match
    # case-insensitively -- a lowercase-only check silently returned "unknown"
    # for every state and would have made wait() spin until its timeout
    low = text.lower()
    for state in ("complete", "error", "cancelacknowledged", "running", "queued"):
        if state in low:
            return state
    return "unknown"


def wait(name: str, timeout_min: int = 540, poll_s: int = 60) -> str:
    """Poll until the kernel leaves the running/queued state.

    Kaggle caps sessions at 12 h (9 h on TPU), so the default ceiling sits just
    under that -- a run still going at 9 h has hit a limit, not a slow epoch.
    """
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        st = status(name)
        if st in ("complete", "error", "cancelacknowledged"):
            return st
        time.sleep(poll_s)
    return "timeout"


# The prep kernel's output is ~9 GB of .npy and this container has ~3.5 GB
# free. Pulling all of it has now filled the disk to 100% twice -- once badly
# enough to threaten checkpointing and git. The arrays exist so a Kaggle
# TRAINING kernel can mount them; only logs, stats and checkpoints ever need to
# travel. So the fetch is filtered by default and the filter is opt-out.
SMALL_FILES = r"\.(json|csv|log|txt|md|pt|png)$"


def output(name: str, dest: str, pattern: str = SMALL_FILES) -> None:
    p = pathlib.Path(dest) / name
    p.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", slug(name), "-p", str(p)]
    if pattern:
        cmd += ["--file-pattern", pattern]
    else:
        print("windml WARNING: unfiltered fetch; kernel outputs can be many GB")
    r = _run(cmd, check=False)
    print((r.stdout + r.stderr).strip())
    files = sorted(f for f in p.rglob("*") if f.is_file())
    total = sum(f.stat().st_size for f in files)
    print(f"windml fetched={len(files)} files, {total/1e6:.1f} MB -> {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "status", "wait", "output", "list"])
    ap.add_argument("name", nargs="?", choices=sorted(KERNELS) + [None])
    ap.add_argument("--dest", default="artifacts/kaggle")
    ap.add_argument("--timeout-min", type=int, default=540)
    ap.add_argument("--file-pattern", default=SMALL_FILES,
                    help="regex of filenames to fetch; '' fetches everything "
                         "(which for prep_era5 is ~9 GB and will fill the disk)")
    a = ap.parse_args()

    if a.action == "list":
        for k, v in KERNELS.items():
            acc = "GPU" if v.get("gpu") else ("TPU" if v.get("tpu") else "CPU")
            print(f"  {k:14s} {acc:3s}  {slug(k):38s} {v['note']}")
        return
    if not a.name:
        sys.exit("a kernel name is required for this action")

    if a.action == "push":
        push(a.name)
    elif a.action == "status":
        status(a.name)
    elif a.action == "wait":
        print(f"windml final_state={wait(a.name, a.timeout_min)}")
    else:
        output(a.name, a.dest, a.file_pattern)


if __name__ == "__main__":
    main()

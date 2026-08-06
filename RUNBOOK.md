# RUNBOOK — RT2021 pretraining on the local GPU box

**Audience: the orchestrating agent on the Mac.** You have SSH to a Linux box with a
GTX 1080. The cloud agent that wrote this has neither SSH nor disk, so you own execution.

Every script here is non-interactive and idempotent — safe to re-run after any failure.
They emit `windml KEY=VALUE` lines and finish with either `RESULT OK` or
`RESULT FAIL <reason>`. **Parse that last line**; exit status matches it. Do not infer
success from the absence of errors.

## Why this exists

Rasp & Thuerey (2021) hold the 5.625° WeatherBench record: **z500 RMSE 268 m²/s² at
3 days**. That number is *CMIP6-pretrained*. Their ERA5-only model scores **314**. We are
reproducing both, in that order — 314 first, because hitting it is what proves the
reimplementation is faithful before any pretraining compute is spent.

## Order of operations

### 1. Setup and the Pascal gate — minutes

```bash
git clone https://github.com/andrewnakas/windplayground-.git   # or git pull
cd windplayground-
git checkout claude/wind-forecast-ml-research-veu6d5
bash scripts/setup_local_gpu.sh
```

**This can fail in a way that matters.** Recent PyTorch wheels dropped the `sm_61`
kernels a GTX 1080 needs, and the failure is quiet — torch imports, `cuda.is_available()`
returns True, then real kernels misbehave. The script runs an actual CUDA matmul and
checks `sm_61` is in the compiled arch list. If it reports the arch is missing, reinstall
from an older index and re-run:

```bash
uv pip install --python .venv-gpu/bin/python \
  --index-url https://download.pytorch.org/whl/cu126 torch
```

Expect `RESULT OK` plus `windml cuda_matmul=ok`. Anything else, stop here.

Override the 150 GB disk floor with `WINDML_MIN_DISK_GB=<n>` only if you have
deliberately chosen a smaller variable subset (see below).

### 2. ERA5 cache — ~1 h, ~9 GB

```bash
.venv-gpu/bin/python scripts/build_cache.py --variable-set rt2021
```

Streams the WeatherBench-2 zarr and adds analytic TOA solar radiation (computed, not
downloaded — see `src/windml/data/solar.py`). Resumable per year.

### 3. ERA5-only training — the fidelity gate

```bash
.venv-gpu/bin/python scripts/train.py --config configs/rt2021_72h.yaml --auto-resume
.venv-gpu/bin/python scripts/evaluate.py \
  --ckpt artifacts/checkpoints/rt2021_72h/best.pt --name rt2021_72h --split 2017_2018
```

**Gate: z500 @ 72 h should land near 314**, scored on **2017–2018** (their test period,
not our usual 2020 — mixing the two makes the comparison meaningless).

- Within ~5% of 314 → the copy is faithful. Proceed.
- Much worse → something is wrong in the reimplementation. **Do not start the 154 GB
  download to paper over it.** Report the number back.

Then push the checkpoint and results:
```bash
git add artifacts/checkpoints/rt2021_72h artifacts/results && git commit && git push
```

### 4. CMIP6 download — only after step 3 passes

Sizes are measured, not estimated (via `Content-Range`; a plain HEAD reveals nothing):

| variable | 5.625° |
|---|---|
| geopotential | 43.2 GB |
| temperature | 41.8 GB |
| specific_humidity | 45.0 GB |
| u_component_of_wind | 12.0 GB |
| v_component_of_wind | 12.0 GB |
| **total** | **~154 GB** |

**How much disk you actually need depends on the pipeline shape, not the download.**
Read from the zip headers: members are 5-year netCDF chunks starting 1850, deflate at
1.19× (1.31 GB → 1.56 GB), ~33 chunks ≈ 165 years. Extracted, the 154 GB becomes ~183 GB.
But we keep only 7 pressure levels at fp16, so the *final training array is ~35 GB* —
roughly 8× smaller than the extracted netCDF.

| approach | peak disk |
|---|---|
| download all → extract all → convert | ~370 GB |
| per-variable, delete each zip after converting | ~135 GB |
| **per-chunk: extract one 1.6 GB chunk, convert, delete** | **~85 GB** |

Steady state once built is only **~44 GB** (35 GB CMIP + 9 GB ERA5). So **~100 GB free is
enough** with the per-chunk path, which is what `build_rt_cache.py` does. Do not extract
the whole archive first; that needs 4× the disk and buys nothing.

One thing to check on the first extract: u/v are 12 GB against z/T/q's 42–45 GB, same
time span and grid. They must differ in vertical levels or dtype. The build script
reports the actual level list per variable — confirm all 7 of the paper's levels are
present for every variable before trusting the pretraining inputs.

```bash
.venv-gpu/bin/python scripts/fetch_cmip.py --dest /data/cmip --dry-run   # sizes only
.venv-gpu/bin/python scripts/fetch_cmip.py --dest /data/cmip
```

It sizes the job and refuses up front if the disk cannot hold it, rather than dying
100 GB in. Resumes via HTTP Range; just re-run after any interruption. Expect ~13 s of
server latency before the first byte and 6–28 MB/s, so budget hours, not minutes.

**If disk is tight**, `--vars` is the lever. Dropping `specific_humidity` saves 45 GB and
still covers both scored pressure-level targets. `--vars geopotential,temperature` is
85 GB and the minimum that makes sense.

Note z/T/q are ~3.6× the size of u/v, which probably means they carry more vertical
levels. Worth confirming after the first extract — it determines how many of the paper's
7 levels each variable actually supplies.

### 5. Pretrain, then fine-tune

```bash
.venv-gpu/bin/python scripts/train.py --config configs/rt2021_pretrain.yaml --auto-resume
.venv-gpu/bin/python scripts/train.py --config configs/rt2021_72h.yaml \
  --run-name rt2021_72h_pre \
  --init-ckpt artifacts/checkpoints/rt2021_pretrain/best.pt --auto-resume
```

CMIP6 has **no** 2m temperature and no precipitation, so pretraining sees 111 input
channels and fine-tuning wants 117. `src/windml/models/grow.py` zero-fills the new stem
columns, which makes the grown model bit-identical to the pretrained one at step 0 — the
transfer is exact, not approximate. That is automatic; nothing to configure.

Pascal is fp32 in practice, so expect ~1.5–2× the paper's ~1 day on a GTX 2080. The box
has no session limit, so let it run; `--auto-resume` restores exact optimizer state after
any interruption.

**Target: z500 @ 72 h ≈ 268 on 2017–2018.**

## Can this all move to Kaggle TPUs instead?

**Uploading the 154 GB is unnecessary** — only the *processed* arrays need to travel:
~35 GB CMIP + ~9 GB ERA5 = **~44 GB**. Build on the box, upload the compact result. The
binding constraint is home upload bandwidth, not Kaggle (~5 h at 20 Mbit/s, ~1 h at
100 Mbit/s). Check Kaggle's current per-dataset cap yourself rather than trusting a
number from me; it has moved before.

**Do not port this model to data-parallel TPU.** 6.4M parameters on a 64×32 grid is tiny,
a v3-8 is wildly overpowered, the bottleneck becomes data loading rather than compute,
and RT2021's BatchNorm means per-core batch statistics stop matching the paper's batch-32
single-device recipe. The `torch_xla` work would buy little and risk a subtly different
model.

**The TPU quota is worth spending a different way: one independent model per core — 8
seeds at once.** This is strictly better here:

- Each core is an independent model with its own batch, so cross-replica BatchNorm never
  arises and the recipe stays *exactly* the paper's.
- No large-batch LR retuning.
- It produces the **ensemble** directly, and ensembling is the one lever already measured
  in this repo: `avg5` beat its best member by 5.7–7.3% with zero fitted parameters.
- `xmp.spawn` with a per-core seed is far less code than a correct data-parallel port.

So: **1080 box → pretraining + the ERA5 fidelity run; Kaggle TPU → the 8-seed ensemble;
Kaggle GPU → per-lead fine-tunes.**

## Reporting rules

These are not stylistic — they are what keeps the numbers meaningful.

- **2017–2018 and 2020 are different scoreboards.** Never mix them in one table.
- **The blend's z500 of 98 is not comparable to 268.** It blends other groups' 0.25°
  forecasts; 268 is a from-scratch 5.625° model. They stay in separate tables.
- Report what the run produced, including a miss. A near-miss reported as a win is worse
  than no result.

## Failure modes

| symptom | cause | action |
|---|---|---|
| `RESULT FAIL ... no sm_61 kernels` | torch wheel lacks Pascal | reinstall from cu126 index |
| `RESULT FAIL torch cannot see the GPU` | driver/CUDA mismatch | check `nvidia-smi` works as this user |
| `RESULT FAIL need XGB but only YGB free` | disk | `--vars` subset, or a bigger disk |
| `RESULT FAIL ... did not verify` | truncated zip | delete that one `.zip` and re-run |
| download stalls at 0 B for ~13 s | normal server latency | wait |
| CUDA OOM | batch too large for 8 GB | halve `batch_size` in the config |
| z500 far from 314 at step 3 | reimplementation bug | stop; report the number |

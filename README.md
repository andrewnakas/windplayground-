# windplayground — global wind forecasting with ML

Recreations of the leading global ML weather-model families at CPU scale, evaluated
WeatherBench-style on wind, plus an ensemble that **beats every published frontier model
on 2020 wind at the evaluated resolution**.

- **[`REPORT.md`](REPORT.md)** — literature review: the top papers, entities, and
  benchmarks in ML wind forecasting, and what we found.
- **[`RESULTS.md`](RESULTS.md)** — full tables and figures (regenerate with
  `scripts/make_report.py`).

## Headline result

Latitude-weighted RMSE on 2020, 00/12 UTC inits, ERA5 truth at 64×32 (5.625°) — every
row scored by the *same* pipeline, including the competitors' own published forecasts as
regridded by WeatherBench 2.

| 10m wind speed RMSE (m/s) | 24 h | 72 h | 120 h |
|---|---|---|---|
| **avg5** (GraphCast+Pangu+HRES+GenCast-mean+FuXi) | **0.361** | **0.791** | **1.380** |
| avg4 (same without FuXi) | 0.366 | 0.796 | 1.386 |
| blend3 (fit on 2018, affine per variable/lead) | 0.369 | 0.803 | 1.421 |
| FuXi | 0.383 | 0.853 | 1.470 |
| GraphCast | 0.389 | 0.858 | 1.479 |
| GenCast (ensemble mean) | 0.393 | 0.862 | 1.474 |
| Pangu-Weather | 0.416 | 0.909 | 1.551 |
| ECMWF HRES | 0.507 | 0.998 | 1.664 |
| our U-Net (1.06M params, 4 CPU cores) | 1.135 | 2.237 | 2.759 |
| persistence | 2.545 | 3.110 | 3.199 |

The ensemble is **−5.7% / −7.3% / −6.1%** against the best individual model at
24/72/120 h, with **zero fitted parameters**. Removing HRES — the weakest member by a
wide margin — makes it *worse* (0.814 at 72 h): it is the only physics-based member, so
its errors are decorrelated from the ML models', and that is worth more than its skill
deficit.

What did *not* work, and why it matters: a 1M-parameter U-Net corrector trained on
GraphCast's 2018 forecasts **degraded** its 2020 forecasts (0.971 vs 0.858 at 72 h),
while a 2-parameter affine recalibration of the same model was neutral (0.855).
WeatherBench's 2018 and 2020 GraphCast forecasts come from different model versions, and
high-capacity post-processing latches onto version-specific error structure. Skill comes
from *model diversity*, not from correcting one already well-calibrated model.

## What's here

| Module | Recreates | Family |
|---|---|---|
| `models/unet.py` | Weyn/Rasp-era CNN baselines | convolutional encoder–decoder |
| `models/vit.py` | **Stormer** (Pangu/Swin lineage) | vision transformer |
| `models/afno.py` | **FourCastNet** | spectral neural operator (AFNO) |
| `models/graph.py` | **GraphCast**/Keisler | icosahedral-mesh GNN |
| `train/corrector.py`, `scripts/blend.py` | GenCast's probabilistic goal, cheaply | learned post-processing / superensemble |

Shared training recipe, all from the GraphCast/Stormer playbook: two-frame input,
residual targets scaled by the 6 h-difference std, latitude-weighted MSE, AdamW +
cosine, and autoregressive rollout fine-tuning (K=2→4, worth 8–11% here).

## Quickstart

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m pytest                        # 23 unit tests
.venv/bin/python scripts/build_cache.py           # ~4 GB ERA5 cache (resumable)
.venv/bin/python scripts/fetch_competitors.py     # published WB2 forecasts at 64x32

.venv/bin/python scripts/evaluate.py --model persistence      # baselines
.venv/bin/python scripts/train.py --config configs/unet.yaml --auto-resume
.venv/bin/python scripts/evaluate.py --ckpt artifacts/checkpoints/unet/best.pt
.venv/bin/python scripts/blend.py --equal-weights \
    --members graphcast pangu hres gencast_mean_2020 --name avg4
.venv/bin/python scripts/make_report.py           # regenerate RESULTS.md
```

Smoke test in <10 min: `build_cache.py --years 2020 2020`, then `train.py
--config configs/unet.yaml --max-steps 200`, then `evaluate.py --init-stride 64`.

Long runs: `scripts/run_queue*.sh` chain jobs sequentially (on 4 cores, two concurrent
jobs each run more than twice as slow). Every stage takes `--auto-resume` and continues
from `last.pt` at the exact LR-schedule position after an interruption.

## Scaling up (GTX 1080 / free Kaggle GPU)

`configs/vit_medium.yaml` trains at 128×64 (2.8125°) with a 10–30M-param model, and
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) runs it inside Kaggle's
free GPU quota (~30 h/week, 12 h sessions). Checkpoints are device-portable, so a model
trained there is scored by the same `evaluate.py` here.

Pascal note (GTX 10-series): install a torch build that still ships `sm_61` kernels —
`cu118`/`cu126` wheels of torch ≤ 2.6; `cu128`+ dropped Pascal. Training is
device-agnostic (`--device auto|cpu|cuda`) and fp32 by default, since Pascal has no fast
half-precision path.

## Live forecasts, and adding WeatherNext

The viewer carries a **live multi-model mean** alongside its members: ECMWF AIFS
(operational ML) and NOAA GFS (operational physics), refreshed every 6 h by
[`.github/workflows/live-wind.yml`](.github/workflows/live-wind.yml) and averaged by
`scripts/blend_live.py`. That pairing is deliberate — the sharpness results in REPORT.md
show ML models are blurred at long leads and physics models are not, so one of each is
the complementary combination.

`blend_live.py` refuses to average members whose init times differ. Filenames are stable
and overwritten in place, so a failed fetch leaves a valid-looking file behind, and
averaging two different cycles is not an ensemble.

GenCast, GraphCast, Pangu and FuXi are **not** available live — WeatherBench-2 ships
their 2020 archives, not an operational feed (probed; see the note in
`scripts/fetch_dynamical.py`). If you have **Google WeatherNext** access, that is the
frontier-grade member this blend is otherwise missing, and you can add it from your own
machine with no key handling at all:

```bash
gcloud auth application-default login
pip install google-cloud-bigquery
python scripts/wn_export_local.py --probe  --project YOUR_PROJECT   # schema, free
python scripts/wn_export_local.py --export --project YOUR_PROJECT --table PROJ.DS.TABLE
git add docs/data && git commit -m "weathernext live" && git push
```

It uses your existing Application Default Credentials, dry-runs every query and refuses
anything over `--max-gb` (your project is the one billed), aggregates to the 180×90 grid
*inside* BigQuery so only ~0.1 MB per lead comes back, and prints the column mapping it
inferred rather than guessing silently. `scripts/fetch_weathernext.py` is the same job
for a container that has credentials in the environment instead.

**If the grant is on a service account, add `--impersonate`.** WeatherNext access is
usually attached to a service account rather than to your login, and the two are
different principals — so the same command that works for the service account 403s for
you, which reads as "the access didn't work" when it is really "wrong principal". A
service-account email like `631486859154-compute@developer.gserviceaccount.com` is a
**grantee identifier, not a secret**: it is safe to paste, and it authenticates nothing
on its own.

```bash
python scripts/wn_export_local.py --probe --project YOUR_PROJECT \
    --impersonate SERVICE_ACCOUNT_EMAIL
```

That needs `roles/iam.serviceAccountTokenCreator` on the service account, granted to
you. Every run prints the principal it is acting as *before* it prints what it can see,
so an empty listing is diagnosable rather than a dead end, and a 403 says which of the
two grants is missing.

WeatherNext's schema is nested — position is a GEOGRAPHY point and every variable hangs
off a REPEATED `forecast` record — so the exporter UNNESTs it, reads lat/lon with
ST_Y/ST_X, and fetches all six leads in one query rather than re-scanning the array six
times. Use the `…_mean` table; the 64-member one costs ~64× for a field that gets
averaged anyway. `--match-live` pins the export to the init AIFS and GFS already carry,
because `blend_live.py` refuses to average different cycles.

**[`RUNBOOK-mac.md`](RUNBOOK-mac.md)** has the whole sequence end to end.

## Honest scope

Our from-scratch models train on 4 CPU cores at 5.625°; GraphCast et al. train at 0.25°
on hundreds of GPU/TPU-days. Those rows are **not** a like-for-like comparison of
architectures — they are the reference ceiling. The like-for-like comparison is against
published 5.625° results (Rasp & Thuerey 2021 ResNet: z500 RMSE 268/499 at 3/5 d) and
against our own baselines. The blend result *is* like-for-like: it beats those models on
their own published forecasts, at the resolution everything is scored on.

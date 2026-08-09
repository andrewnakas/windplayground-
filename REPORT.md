# Global Wind Forecasting with Machine Learning — Literature Review

*Survey date: August 2026. This report identifies the top papers, entities, and benchmarks for
ML-based global wind forecasting, and maps each leading model family to the small-scale
recreation implemented in this repository.*

---

## 1. The benchmark landscape

### WeatherBench 2 — the canonical global benchmark

**WeatherBench 2** (Rasp et al., *JAMES* 2024, Google Research + ECMWF) is the standard
evaluation framework for data-driven global weather models. Everything in this repo follows its
conventions:

- **Ground truth:** ERA5 reanalysis (ECMWF), 1959–present, 0.25° native resolution, with
  official conservatively-regridded versions at 240×121, 128×64, and **64×32 (5.625°)** —
  the grid used here.
- **Wind headline variables:** 10m u/v wind components (`u10`, `v10`), derived **10m wind
  speed**, and 850hPa u/v wind (`u850`, `v850`). Wind is a headline because it drives the
  renewable-energy use case (wind power ∝ v³ in the operating range).
- **Metrics:** latitude-weighted RMSE and ACC (anomaly correlation vs. a smoothed
  climatology) for deterministic forecasts; **CRPS** and spread-skill ratio for ensembles.
- **Baselines:** ECMWF IFS **HRES** (the best physics-based deterministic model), IFS **ENS**
  (the 50-member physics ensemble), persistence, and climatology.
- **Protocol:** test on held-out recent years (2020 is standard), initialize from analysis
  states, evaluate at 6h-multiple lead times out to 10–15 days.

A crucial, under-advertised fact exploited in this project: the WB2 public bucket
(`gs://weatherbench2`) ships the **actual forecast outputs** of the leading models
(GraphCast, GenCast incl. full ensemble, Pangu-Weather, FuXi, Aurora, NeuralGCM, HRES, ENS)
for 2020 — and for GraphCast also 2018 — pre-regridded to 64×32. This makes head-to-head
scoring (and learned post-processing) of the real frontier models possible at small scale.

### Other relevant benchmarks

- **WeatherBench 1** (Rasp et al. 2020): the original 5.625° benchmark; its published
  leaderboard numbers are the directly comparable anchors for our from-scratch models.
- **ExtremeWeatherBench** (2025/26): high-impact events, including damaging wind.
- **Wind-power-specific:** GEFCom (energy forecasting competitions) and the Baidu KDD Cup
  SDWPF turbine dataset — station/farm scale rather than global; out of scope here but the
  natural downstream application.

---

## 2. Entities and models at the top of the wind benchmarks (mid-2026)

### Google DeepMind — GraphCast and GenCast

- **GraphCast** (Lam et al., *Science* 2023). Multi-mesh **graph neural network**:
  encode grid → refined icosahedral mesh, 16 rounds of message passing across mesh scales,
  decode back to grid. 37M params, 0.25°, trained on ERA5 1979–2017 (~4 weeks on 32 TPUs).
  Beat HRES on ~90% of 1380 targets — the deterministic breakthrough. Open weights.
  Key training tricks adopted here: **residual (Δ-state) targets scaled by per-variable
  6h-difference std**, two input frames (t, t−6h), and **autoregressive rollout fine-tuning**.
- **GenCast** (Price et al., *Nature* 2025). **Conditional diffusion model** generating
  50-member ensembles at 0.25°, 12h steps. Beats the ECMWF ENS on **97.4% of targets**,
  including 10m wind CRPS and regional wind-power aggregates — the current state of the art
  for probabilistic wind. Open weights/code (JAX).

### Huawei — Pangu-Weather

Bi et al., *Nature* 2023. **3D Earth-specific Swin transformer** treating the pressure-level
stack as a 3D volume; hierarchical temporal aggregation (1h/3h/6h/24h models) to reduce
rollout error accumulation. First ML model to beat operational IFS deterministically on
upper-air variables. Open ONNX weights.

### ECMWF — AIFS

The first ML forecast system run **operationally** by a weather center (v1 2024–25, ENS
variant 2025). GNN-transformer hybrid on an octahedral reduced-Gaussian grid, trained on
ERA5 + operational analyses. Notably strong on tropical cyclone tracks; competitive 10m wind.

### NVIDIA — FourCastNet / SFNO

Pathak et al. 2022; Bonev et al. ICML 2023. **Fourier/spherical-harmonic neural operators**:
token mixing in the frequency domain (AFNO), later on the sphere (SFNO) for
rotation-equivariant, stable long rollouts. The efficiency lineage — FourCastNet was the
first 0.25° ML model, ~45,000× faster than IFS per forecast.

### Microsoft — Aurora

Bodnar et al., *Nature* 2025. 1.3B-param **foundation model** (3D Swin encoder-decoder +
Perceiver) pretrained on >1M hours of heterogeneous atmospheric data, fine-tuned per task.
State of the art on 0.1° operational forecasting and strong on wind at the surface; also
air quality and ocean waves. Demonstrates the pretrain→finetune paradigm for weather.

### Stormer (UCLA/CMU, NeurIPS 2024)

Nguyen et al. A deliberately **simple ViT** with three key ingredients — weather-specific
embedding, randomized dynamics forecasting (train one model at 6/12/24h intervals), and
pressure-weighted loss — matching GraphCast-class skill at 1.4° with far less compute.
Proof that architecture simplicity + right training recipe ≈ frontier skill; the closest
published analogue to what this repo does at small scale.

### Others at the frontier

- **FuXi / FengWu** (Fudan U. / Shanghai AI Lab): cascaded transformers (FuXi chains
  lead-time-specialized models); FengWu-GHR reached 0.09°. Strong RMSE at long leads.
- **NeuralGCM** (Google, *Nature* 2024): differentiable dynamical core + learned physics —
  the hybrid approach; best-in-class ensemble calibration at 1.4°.
- **Jua** (commercial, Zurich): EPT-2/EPT-2e; 2026 marketing benchmarks claim leads over
  ENS and AI rivals on **European wind and solar** — the commercial wind-energy frontier
  (numbers not independently verified on WB2).
- **Keisler 2022**: the 1° GNN (6.7M params) that started the GNN line — the closest
  historical precedent for "small model, real skill."

### Published anchors at our 5.625° scale (directly comparable)

| Model | Z500 RMSE 3d/5d (m²/s²) | T850 RMSE 3d/5d (K) |
|---|---|---|
| Persistence | 936 / 1033 | 4.23 / 4.56 |
| Climatology (WB1) | 816 | 3.50 |
| Rasp & Thuerey 2021 ResNet, direct, **ERA5 only** (6.3M params) | **314 / 561** | 1.79 / 2.82 |
| Rasp & Thuerey 2021 ResNet, direct, **CMIP6-pretrained** | **268 / 523** | **1.65 / 2.52** |
| Rasp & Thuerey 2021 ResNet, continuous, CMIP6-pretrained | 284 / 499 | 1.72 / 2.41 |
| IFS T63 | 268 / 463 | 1.85 / 2.52 |
| Operational IFS regridded to 5.625° | 154 / 334 | 1.36 / 2.03 |

(Transcribed from Table 1 of arXiv 2008.08626v2. An earlier version of this table carried
a single "CMIP-pretrained" row reading 268 / 499 and 1.65 / 2.41 — the 3-day figures come
from *Direct (pretrained)* and the 5-day ones from *Continuous (pretrained)*, two
different models. The rows are separated above.)

(Rasp & Thuerey did not train on wind; our tables add u10/v10/wind-speed rows scored
identically, with the published 64×32-regridded GraphCast/HRES forecasts as the reference
competitor rows.)

---

## 3. What we recreate and why

Frontier models train on hundreds of GPU/TPU-days at 0.25° (≈2 TB of data). On this
project's compute (4 CPU cores; optionally a GTX 1080 / free Kaggle GPU), the honest
scientific move — standard since WeatherBench 1 — is to recreate the **architecture
families** at 5.625° with 1–3M params and compare against published numbers at the same
resolution, plus the regridded true-model forecasts.

| This repo | Recreates | Family |
|---|---|---|
| `models/unet.py` | Weyn/Rasp-era CNN baselines | Convolutional encoder-decoder |
| `models/vit.py` | **Stormer** (and the Pangu/Swin transformer lineage) | Vision transformer |
| `models/afno.py` | **FourCastNet** (AFNO token mixing) | Spectral neural operator |
| `models/graph.py` (stretch) | **GraphCast/Keisler** | Multi-mesh GNN |
| corrector + blend (Phase 9) | **GenCast's** probabilistic goal via post-processing | Learned post-processing / superensemble |

Training recipe shared across all models (all from the GraphCast/Stormer playbook):
two-frame input, residual targets scaled by 6h-difference std, latitude-weighted MSE,
AdamW + cosine schedule, autoregressive rollout fine-tuning (K=2→4), fixed 6h step,
evaluation to 120h.

### The "beat them" plan

Recreation alone cannot outskill GraphCast. **Learned post-processing can**: systematic
errors of a frozen forecast model are learnable (the post-processing literature reports
10–40% wind RMSE reductions at stations; grid-scale corrections are smaller but real), and
WB2 publishes GraphCast's own 2018 forecasts to train on and 2020 forecasts to test on.
Target: beat raw GraphCast and HRES on 2020 10m-wind RMSE (and CRPS via ensemble dressing)
at the evaluated 64×32 resolution. A per-lead multi-model blend (GraphCast + Pangu + HRES +
ours) is the second lever — superensembles classically beat every member.

### What we found (see RESULTS.md for the numbers)

**1. Our metrics pipeline reproduces the published ordering.** Scoring the WB2-published
64×32 forecasts with our own code puts GraphCast ahead of Pangu ahead of HRES, with
GenCast's ensemble mean best at long leads — exactly the published ranking. Persistence
(z500 RMSE 923/1019 at 3/5 d) and climatology (803) also land on the WeatherBench-1
reference values (936/1033 and 816). The harness is trustworthy.

**2. At CPU scale, the CNN beats the transformer, the spectral operator, and the GNN.**
On 10m wind speed RMSE at 72 h: U-Net 2.44 (1.06M params) < ViT 2.76 (2.77M) < AFNO 2.82
(2.02M) < mesh-GNN 3.31 (0.72M). This is not evidence against Stormer, FourCastNet, or
GraphCast — it is the expected regime effect. Attention, spectral token-mixing, and
message passing all have weaker inductive biases than convolution and need far more data
and compute before they overtake it; our budget was ~2 epochs of 5.625° ERA5 on 4 CPU
cores. The GNN is additionally handicapped by construction: a single 642-node icosphere
for 2048 grid points is a severe bottleneck, where GraphCast uses a *multi*-mesh (six
refinement levels at once, 40,962 nodes) so that both local and hemispheric interactions
have short paths. Reproducing GraphCast's architecture at 1/50th the mesh and 1/50th the
parameters reproduces its structure, not its skill.

**3. Over-parameterized post-processing does *not* transfer across model versions
(negative result).** A 1M-parameter U-Net corrector trained on GraphCast's published 2018
forecasts *degraded* its 2020 forecasts (10m wind speed RMSE 0.97 vs 0.86 at 72 h). The
2018 and 2020 GraphCast forecast sets come from differently-trained model versions, so a
high-capacity corrector fits version-specific error structure that no longer exists at
test time. This is the post-processing analogue of overfitting to a stale model.

**3b. A weaker, differently-built member still improves the ensemble.** HRES is by far
the worst member on wind (0.998 vs FuXi's 0.853 at 72 h), so dropping it looks like an
easy win — but removing it makes the average *worse* (0.814 vs 0.791). It is the only
physics-based member, so its errors are decorrelated from the ML models', and that
decorrelation is worth more than its individual skill deficit. Ensemble value comes from
diversity, not from member quality alone. (Conversely, adding FuXi — the strongest
individual on wind — moved the average only 0.796 → 0.791, since it errs much like the
other ML members.)

**4. Low-parameter multi-model blending *does* transfer — and beats every frontier
model.** A per-(variable, lead) affine least-squares blend of GraphCast + Pangu + HRES,
with weights fit on 2018 and applied unchanged to 2020, beats the best individual
competitor at every lead. Three parameters per (variable, lead) generalize across model
versions where a million do not — the same distribution shift that broke finding 3
leaves the blend untouched.

Better still, a plain **equal-weight average needs no fitting at all**, so members that
only exist for the test year (GenCast, FuXi) can join. The five-member average
(GraphCast + Pangu + HRES + GenCast-mean + FuXi) is our best forecast:

| 10m wind speed RMSE (m/s) | 24 h | 72 h | 120 h |
|---|---|---|---|
| **avg5** | **0.361** | **0.791** | **1.380** |
| best individual (FuXi) | 0.383 | 0.853 | 1.470 |
| GraphCast | 0.389 | 0.858 | 1.479 |

That is **−5.7% / −7.3% / −6.1%** against the best single model at 24/72/120 h, with
zero fitted parameters. *Caveat worth stating:* we compared a handful of member
combinations on the test year, so picking avg5 over avg4 (0.791 vs 0.796) is selection
on the test set and that 0.6% gap is within noise. The robust claim is the family
result — a 4–5 member multi-model average beats every individual model by 5–7% — not
the precise ranking among averages.

### Chasing the 5.625° anchor: how far we got, and what actually blocks it

**Correction first, because it changes what the target even is.** Earlier drafts of this
section quoted Rasp & Thuerey's **268** as the number a from-scratch ERA5 model should
reach. It is not. Reading Table 1 of arXiv 2008.08626v2 directly:

| their model | z500 @ 3d / 5d |
|---|---|
| Direct, **ERA5 only** | **314 / 561** |
| Direct, **CMIP6-pretrained** | 268 / 523 |
| Continuous, CMIP6-pretrained | 284 / 499 |

268 requires pretraining on ~150 years of CMIP6 MPI-ESM-HR. The apples-to-apples target
for a model trained only on ERA5 is **314**. Every "gap to 268" figure below was
measured against the wrong anchor, and the gap to the right one is proportionally
smaller — though still a miss, so the conclusion does not change, only its size.

Progression, each step a deliberate change:

| model | z500 @3d | what changed |
|---|---|---|
| `unet` (9.5k steps) | 539 | baseline |
| `unet_long` (45k steps) | 455 | 4.7× training |
| `unet_long_ft2` | 435 | + K=2 rollout curriculum |
| `unet_long_ft4` | 412 | + K=4 |
| `levels72` | 394 | + vertical inputs, direct 72h target (see ablation below) |
| `anchor72` | 387 | + 6.6M params, z500-weighted loss |
| `resnet72` | 438 | full-resolution ResNet, 0.5M — too small, see below |
| `resnet72_big` | **378** | full-resolution ResNet, 1.7M, equal wall-clock |
| *target (ERA5-only)* | *314* | |
| *their pretrained number* | *268* | needs 150 y of CMIP6 |

**We did not reach it — the best from-scratch z500 is 378.5, 21% above the 314 ERA5-only
anchor** (41% above the pretrained 268, which was never the right comparison). Two of
these steps are worth dwelling on because they were my main hypotheses. Capacity plus
loss weighting mostly failed: going from 2.95M to 6.6M parameters *and* weighting the
loss 20× onto z500 together bought 1.6%. Architecture did not: dropping to a 1.7M
full-resolution ResNet at the same wall-clock bought another 2.3% and produced the best
number here, which is analysed in full below.

Benchmarking the architecture finally explained why. Their model is a
**fully-convolutional ResNet that never pools** — it holds 32×64 through 19 residual
blocks. Our U-Net pools to 8×16 at the bottleneck, discarding much of a field that is
only 2048 points to begin with. Measured on this machine:

| architecture | s/step (batch 16) | 14k steps |
|---|---|---|
| U-Net (pools 4×), 6.6M params | 1.1 | 4.3 h |
| full-res ResNet, 0.5M params (width 64, 6 blocks) | 0.84 | 3.3 h |
| full-res ResNet, 1.7M params (width 96, 10 blocks) | 3.1 | 12 h |
| **their model** (19 blocks, 128 filters) | 10.6 | **41 h** |

(Rates measured uncontended. An earlier draft of this section quoted 10.6 s/step for the
1.7M model and ~4 days for theirs; those came from a benchmark run while another job had
the CPU, and overstated the wall by ~3×. Their architecture is still out of reach at
41 h, but the middle rows are affordable and the corrected numbers are above.)

From this I predicted that **resolution was the binding constraint** — that our U-Net
stalled at 387 mainly because it pools 32×64 down to 8×16. `configs/resnet72.yaml` tested
it directly: same inputs, same direct-72h target, same loss weighting, same wall-clock as
`anchor72`, but spent on a small full-resolution model (0.5M params) instead of a large
pooled one (6.6M).

**Which half of `levels72` did the work?** That row bundles two changes — vertical
inputs and a direct 72 h target — and `direct72` separates them by applying only the
direct target to the plain 8-channel set:

| model | channel set | target | steps | z500 @3d |
|---|---|---|---|---|
| `direct72` | core (8 ch) | direct 72 h | 20k | 455.6 |
| `unet_long` | core (8 ch) | iterative 6 h | 45k | 454.9 |
| `unet_long_ft4` | core (8 ch) | + K=4 rollout fine-tune | 45k + | 411.7 |
| `levels72` | levels (vertical) | direct 72 h | 14k | 393.7 |

The budgets are not equal — `direct72` got 20k steps against `unet_long`'s 45k — so this
is weaker than the resolution comparison above and is reported as such. What it does show
is that the direct target alone lands on top of the iterative base model (455.6 vs 454.9)
at under half the steps, and clearly behind the rollout-fine-tuned version. So the ~15%
that `levels72` gained over `unet_long` is attributable mainly to the **vertical inputs**,
not to the direct target, and the progression table above should be read that way.
Rollout fine-tuning is separately worth ~10% on the iterative path, and the direct models
never received an equivalent refinement.

**The test appeared to refute the prediction** — and was itself later overturned; the
resolved version is the table two blocks down, and this one is kept because the sequence
is the point.

| model | architecture | params | z500 @3d |
|---|---|---|---|
| `anchor72` | U-Net, pools to 8×16 | 6.6M | **387** |
| `levels72` | U-Net, pools to 8×16 | 2.95M | 394 |
| `resnet72` | ResNet, full 32×64 | 0.5M | **438** |

Full resolution came out *worse*, not better. At a fixed CPU budget, parameters behind a
pooling bottleneck beat resolution — the opposite of what I expected.

The honest caveat at the time: this equalized *wall-clock*, not *capacity*, and the
ResNet had 13× fewer parameters, so it did not prove resolution irrelevant — only that
it was not worth that much capacity. `configs/resnet72_big.yaml` (1.7M params, full
resolution) was the remaining disentangler.

**And the disentangler reversed the reversal.** `resnet72_big` scores z500 **378.5** —
the best from-scratch number in this repo, beating the 6.6M pooled `anchor72` with 3.9×
fewer parameters:

| model | architecture | params | wall-clock | z500 @3d | t850 @3d |
|---|---|---|---|---|---|
| `resnet72_big` | ResNet, full 32×64 | 1.7M | 4.27 h | **378.5** | 1.988 |
| `anchor72` | U-Net, pools to 8×16 | 6.6M | 4.36 h | 387.4 | **1.870** |
| `levels72` | U-Net, pools to 8×16 | 2.95M | 2.22 h | 393.7 | 1.992 |
| `resnet72` | ResNet, full 32×64 | 0.5M | 1.89 h | 437.6 | 2.230 |

The comparison is controlled where it matters: `resnet72_big` and `anchor72` share the
variable set, two-frame inputs, batch size 16, LR 4e-4, the direct 72 h target and — the
part that would otherwise confound this — **identical channel loss weights** (z500 20,
t850 6). Wall-clock differs by 2%.

So the sequence was: I predicted resolution binds → `resnet72` appeared to refute it →
`resnet72_big` shows the refutation was a **capacity artifact**, 0.5M being simply too
small to represent the field regardless of grid. At comparable capacity, full resolution
wins. Note also that `anchor72` weighted z500 at 20× and *still* lost on z500 to a model
with a quarter of its parameters, which makes the architectural reading harder to escape.

Not a clean sweep, and reporting it as one would be wrong: **`anchor72` wins t850**
(1.870 vs 1.988) under the same loss weights. Pooling costs the most on the variable with
the sharpest gradients (geopotential) and costs little on the smoother one.

The earlier claim in this report — "at a fixed CPU budget, capacity buys more than
resolution" — was measured on the 0.5M model and does not survive. The corrected claim is
narrower and the opposite in sign: **at comparable capacity and equal wall-clock, keeping
the full 32×64 grid beats pooling for z500.** That is also the architecture Rasp &
Thuerey use, which is a point in favour of the faithful reproduction rather than more
U-Net tuning.

What this leaves: every lever predicted to close the gap — more capacity, a loss
focused on z500, a direct target, full resolution — delivered little or backfired. The
gap is not explained by anything measured here.

**What was actually different, found only after the fact.** Pulling the paper and the
author's `src/networks.py` side by side against our configs, the models were not
near-variants of each other at all:

| | Rasp & Thuerey | ours (`anchor72`) |
|---|---|---|
| input channels | **117** (5 vars × 7 levels + t2m + precip + TOA solar, at t/t−6h/t−12h, + 3 constants) | 25–49 |
| output channels | **3** (z500, t850, t2m) | 20 |
| training years | 1979–2015 | 1979–2017 |
| test years | **2017–2018** | 2020 |

So this was never a compute deficit being papered over — it was a different model fed
different inputs, scored on a different period. That also means our 378.5 and their 314
are not strictly comparable numbers, which is its own reason the "gap" was never going
to be interpretable.

The faithful reproduction is now built (`src/windml/models/rt_resnet.py`, 6,355,587
parameters against their ~6.3M) and runs on Kaggle GPU against the 2017–2018 split.
**Its gate is 314, not 378 and not 268.** Only if that lands does CMIP6 pretraining —
the one remaining difference, and the thing that buys 314 → 268 — become worth the
quota.

Honest bottom line on this goal as of the CPU-only phase: **the anchor was not met**, and
the reason was modelling, not effort. Note also that our best *forecast* — the 5-member
ensemble at z500 98 — is far past both numbers, but that blends other groups' 0.25°
forecasts and is a different achievement entirely; the from-scratch number is 378.5.

### What a real 0.25° WB2 leaderboard entry would take

For scope honesty: ~2 TB ERA5 at 0.25°, a 30–300M-param model, and O(10²–10³) A100-days
(GraphCast: 32 TPU v4 × 4 weeks). Achievable for a lab, not for free-tier compute. The
free-tier path stops at "competitive at coarse resolution + beats frontier models via
post-processing at that scale."

---

## 4. Key references

- Rasp et al., *WeatherBench 2* (JAMES 2024) — [10.1029/2023MS004019](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004019)
- Lam et al., *GraphCast* (Science 2023) — [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
- Price et al., *GenCast* (Nature 2025) — [10.1038/s41586-024-08252-9](https://www.nature.com/articles/s41586-024-08252-9)
- Bi et al., *Pangu-Weather* (Nature 2023) — [10.1038/s41586-023-06185-3](https://www.nature.com/articles/s41586-023-06185-3)
- Nguyen et al., *Stormer* (NeurIPS 2024) — [arXiv:2312.03876](https://arxiv.org/abs/2312.03876)
- Pathak et al., *FourCastNet* (2022) — [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
- Bonev et al., *SFNO* (ICML 2023) — [arXiv:2306.03838](https://arxiv.org/abs/2306.03838)
- Bodnar et al., *Aurora* (Nature 2025) — [arXiv:2405.13063](https://arxiv.org/abs/2405.13063)
- Kochkov et al., *NeuralGCM* (Nature 2024) — [arXiv:2311.07222](https://arxiv.org/abs/2311.07222)
- Rasp & Thuerey, ResNet for WeatherBench (JAMES 2021) — [10.1029/2020MS002405](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002405)
- Keisler, GNN forecasting (2022) — [arXiv:2202.07575](https://arxiv.org/abs/2202.07575)
- ECMWF AIFS — [ecmwf.int/en/about/media-centre/aifs-blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog)
- Jua EPT-2 benchmark claims — [jua.ai/articles/2026-ai-weather-model-benchmarks](https://jua.ai/articles/2026-ai-weather-model-benchmarks/)

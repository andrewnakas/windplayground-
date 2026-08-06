"""Typed config loading from YAML files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"

# WeatherBench 2 public bucket, anonymously readable over HTTPS.
WB2_ERA5_64x32 = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)

# Dynamic (prognostic) channels, in fixed order. Names follow the WB2 dataset.
VARIABLES: list[dict[str, Any]] = [
    {"name": "10m_u_component_of_wind", "short": "u10", "level": None},
    {"name": "10m_v_component_of_wind", "short": "v10", "level": None},
    {"name": "2m_temperature", "short": "t2m", "level": None},
    {"name": "mean_sea_level_pressure", "short": "msl", "level": None},
    {"name": "u_component_of_wind", "short": "u850", "level": 850},
    {"name": "v_component_of_wind", "short": "v850", "level": 850},
    {"name": "temperature", "short": "t850", "level": 850},
    {"name": "geopotential", "short": "z500", "level": 500},
]
CHANNELS = [v["short"] for v in VARIABLES]

# Extra pressure levels, appended AFTER the scored channels so CHANNELS indices
# (and therefore every metric) stay valid. Rasp & Thuerey's WeatherBench ResNet
# sees z and t through the depth of the atmosphere; with a single z level our
# model has no vertical structure to work with, which is a real handicap for
# predicting how z500 evolves.
EXTRA_VARIABLES: list[dict[str, Any]] = [
    {"name": "geopotential", "short": "z250", "level": 250},
    {"name": "geopotential", "short": "z700", "level": 700},
    {"name": "geopotential", "short": "z850", "level": 850},
    {"name": "geopotential", "short": "z925", "level": 925},
    {"name": "temperature", "short": "t250", "level": 250},
    {"name": "temperature", "short": "t500", "level": 500},
    {"name": "temperature", "short": "t700", "level": 700},
    {"name": "u_component_of_wind", "short": "u250", "level": 250},
    {"name": "u_component_of_wind", "short": "u500", "level": 500},
    {"name": "v_component_of_wind", "short": "v250", "level": 250},
    {"name": "v_component_of_wind", "short": "v500", "level": 500},
    {"name": "specific_humidity", "short": "q700", "level": 700},
]

STATIC_VARIABLES = ["land_sea_mask", "geopotential_at_surface"]

# --- Rasp & Thuerey (2021) input set -----------------------------------------
# Their exact 38 fields per time step: geopotential, temperature, u, v and
# specific humidity on 7 pressure levels (35), plus 2m temperature, 6-hourly
# precipitation and TOA incident solar radiation. Stacked over t, t-6h and
# t-12h that is the 114 dynamic channels the paper reports; land-sea mask,
# orography and latitude bring the conv input to 117.
RT_LEVELS = (50, 250, 500, 600, 700, 850, 925)
RT_LEVEL_VARS = (
    ("geopotential", "z"),
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("specific_humidity", "q"),
)

# The three variables the paper's main network predicts. Precipitation gets a
# separate network -- predicting all four together "led to bad predictions for
# all variables" -- so it is deliberately absent here.
RT_TARGETS = ["z500", "t850", "t2m"]

# Ordered so the RT_TARGETS land in the first three slots: the loss, the
# metrics and the model head all index channels positionally, and keeping the
# scored variables first is the same convention VARIABLES already follows.
RT_VARIABLES: list[dict[str, Any]] = (
    [
        {"name": "geopotential", "short": "z500", "level": 500},
        {"name": "temperature", "short": "t850", "level": 850},
        {"name": "2m_temperature", "short": "t2m", "level": None},
    ]
    + [
        {"name": name, "short": f"{abbr}{lev}", "level": lev}
        for name, abbr in RT_LEVEL_VARS
        for lev in RT_LEVELS
        if f"{abbr}{lev}" not in ("z500", "t850")
    ]
    + [{"name": "total_precipitation_6hr", "short": "tp", "level": None}]
)

# Precipitation is standardized by its std WITHOUT subtracting the mean, after
# a log transform, so that zero stays zero. The paper calls this crucial: with
# a raw or mean-centred target the network just predicts zeros.
LOG_TRANSFORM_VARIABLES = frozenset({"tp"})
PRECIP_LOG_EPSILON = 1e-3

# CMIP6 pretraining covers only the pressure-level variables. Verified against
# the WeatherBench data repository (dataserv.ub.tum.de/s/m1524895): under
# CMIP/MPI-ESM/{2.8125deg,5.625deg} only geopotential, temperature,
# u_component_of_wind, v_component_of_wind and specific_humidity exist -- there
# is no 2m_temperature and no precipitation. Pretraining therefore runs on a
# reduced channel set and the surface inputs are grown in at fine-tune time.
CMIP_AVAILABLE_VARIABLES = frozenset(
    {"geopotential", "temperature", "u_component_of_wind",
     "v_component_of_wind", "specific_humidity"}
)


def rt_pretrain_variables() -> list[dict[str, Any]]:
    """The RT2021 channels that CMIP6 can actually supply."""
    return [v for v in RT_VARIABLES if v["name"] in CMIP_AVAILABLE_VARIABLES]


# TOA incident solar radiation is computed from solar geometry rather than read
# from the zarr (see windml/data/solar.py), so it is not in RT_VARIABLES -- but
# it IS one of the paper's 38 per-timestep fields, and it is appended when the
# cache is built. Counting it here is what makes 117 reproducible:
#   (37 stored + 1 computed) x 3 frames + 3 statics = 117.
RT_COMPUTED_FIELDS = 1
RT_INPUT_FRAMES = 3  # t, t-6h, t-12h
RT_N_STATIC = 3  # land-sea mask, orography, latitude


def rt_input_channels(variable_set: str = "rt2021") -> int:
    """Conv input channels for an RT2021-style run: 117 for ERA5, 111 for CMIP."""
    per_frame = len(active_variables(variable_set)) + RT_COMPUTED_FIELDS
    return per_frame * RT_INPUT_FRAMES + RT_N_STATIC


def active_variables(variable_set: str) -> list[dict[str, Any]]:
    """'core' = the 8 scored channels; 'levels' = those plus vertical structure.

    'rt2021' is the Rasp & Thuerey input set; 'rt2021_cmip' is the subset of it
    that the CMIP6 pretraining archive provides.
    """
    if variable_set == "core":
        return VARIABLES
    if variable_set == "levels":
        return VARIABLES + EXTRA_VARIABLES
    if variable_set == "rt2021":
        return RT_VARIABLES
    if variable_set == "rt2021_cmip":
        return rt_pretrain_variables()
    raise ValueError(f"unknown variable_set: {variable_set}")


@dataclass
class DataConfig:
    zarr_url: str = WB2_ERA5_64x32
    grid: str = "64x32"  # cache namespace; "128x64" for the medium (GPU) tier
    variable_set: str = "core"  # "core" (8 scored) or "levels" (+vertical)
    cache_dir: str = str(ARTIFACTS / "data")
    train_years: tuple[int, int] = (1979, 2017)
    val_years: tuple[int, int] = (2018, 2019)
    test_years: tuple[int, int] = (2020, 2020)

    @property
    def stats_path(self) -> Path:
        """Stats live per variable set, since the channel list differs."""
        suffix = "" if self.variable_set == "core" else f"_{self.variable_set}"
        return Path(self.cache_dir) / f"stats{suffix}.json"

    @property
    def channels(self) -> list[str]:
        return [v["short"] for v in active_variables(self.variable_set)]


@dataclass
class ModelConfig:
    name: str = "unet"
    params: dict[str, Any] = field(default_factory=dict)
    two_frame: bool = True  # feed t and t-6h states (GraphCast-style)


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 32
    max_steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    time_budget_hours: float | None = None
    rollout_steps: int = 1  # K: number of autoregressive steps in the loss
    direct_lead_h: int | None = None  # predict this lead in one shot instead
    val_every: int = 1000
    channel_loss_weights: dict[str, float] = field(default_factory=dict)
    device: str = "auto"  # auto | cpu | cuda


@dataclass
class Config:
    run_name: str = "run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        data = DataConfig(**{
            k: tuple(v) if k.endswith("_years") else v
            for k, v in raw.get("data", {}).items()
        })
        model = ModelConfig(**raw.get("model", {}))
        train = TrainConfig(**raw.get("train", {}))
        return cls(run_name=raw.get("run_name", "run"), data=data, model=model, train=train)

    def hash(self) -> str:
        blob = json.dumps(
            {
                "run_name": self.run_name,
                "data": self.data.__dict__,
                "model": self.model.__dict__,
                "train": self.train.__dict__,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

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

STATIC_VARIABLES = ["land_sea_mask", "geopotential_at_surface"]


@dataclass
class DataConfig:
    zarr_url: str = WB2_ERA5_64x32
    cache_dir: str = str(ARTIFACTS / "data")
    train_years: tuple[int, int] = (1979, 2017)
    val_years: tuple[int, int] = (2018, 2019)
    test_years: tuple[int, int] = (2020, 2020)


@dataclass
class ModelConfig:
    name: str = "unet"
    params: dict[str, Any] = field(default_factory=dict)
    two_frame_input: bool = True  # feed t and t-6h states (GraphCast-style)


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
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
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

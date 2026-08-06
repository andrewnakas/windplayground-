"""Model registry: name -> constructor."""
from __future__ import annotations

from typing import Any

import torch


def build_model(name: str, **params: Any) -> torch.nn.Module:
    if name == "unet":
        from windml.models.unet import UNet

        return UNet(**params)
    if name == "resnet":
        from windml.models.resnet import WeatherResNet

        return WeatherResNet(**params)
    if name == "rt_resnet":
        from windml.models.rt_resnet import WeatherResNetRT

        return WeatherResNetRT(**params)
    if name == "vit":
        from windml.models.vit import WeatherViT

        return WeatherViT(**params)
    if name == "afno":
        from windml.models.afno import AFNONet

        return AFNONet(**params)
    if name == "graph":
        from windml.models.graph import MeshGNN

        return MeshGNN(**params)
    raise ValueError(f"unknown model: {name}")

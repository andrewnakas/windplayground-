"""Fully-convolutional ResNet — the Rasp & Thuerey WeatherBench architecture.

Their 5.625 deg model (z500 RMSE 268 at 3 days) is a deep residual stack that
keeps the grid at **full resolution throughout**. Our U-Net instead pools
32x64 down to 8x16 at the bottleneck, which on a grid this small discards a lot
of the field it is being asked to predict — a plausible reason the U-Net line
stalled around 390.

Design follows the paper: 3x3 convolutions, longitude-periodic padding, LeakyReLU,
batch norm, dropout, and residual connections every two convolutions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from windml.models.unet import CircularConv2d


class ResBlock(nn.Module):
    """Two circular convs with a residual connection, per the paper's block."""

    def __init__(self, channels: int, dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        self.conv1 = CircularConv2d(channels, channels, dilation=dilation)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = CircularConv2d(channels, channels, dilation=dilation)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.3)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.norm1(self.act(self.conv1(x))))
        h = self.drop(self.norm2(self.act(self.conv2(h))))
        return x + h


class WeatherResNet(nn.Module):
    """Full-resolution residual stack.

    `dilations` cycles the dilation rate across blocks so the receptive field
    reaches hemispheric scale without pooling: at 5.625 deg a plain 3x3 stack
    of depth d sees only ~d*5.6 degrees, too local for 3-day flow evolution.
    """

    def __init__(
        self,
        in_channels: int = 49,
        out_channels: int = 20,
        width: int = 96,
        n_blocks: int = 10,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2),
    ):
        super().__init__()
        self.stem = CircularConv2d(in_channels, width)
        self.blocks = nn.ModuleList([
            ResBlock(width, dropout, dilation=dilations[i % len(dilations)])
            for i in range(n_blocks)
        ])
        self.head = nn.Conv2d(width, out_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

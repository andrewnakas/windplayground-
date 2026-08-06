"""U-Net baseline (Weyn/Rasp-era CNN lineage).

Longitude is periodic, so all convs pad circularly in the x direction and with
zeros in latitude. Three resolution levels: 32x64 -> 16x32 -> 8x16.
"""
from __future__ import annotations

import torch
from torch import nn


class CircularConv2d(nn.Module):
    """Conv2d with circular padding in longitude (last axis), zeros in latitude."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = torch.nn.functional.pad(x, (0, 0, self.pad, self.pad))
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            CircularConv2d(in_ch, out_ch),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            CircularConv2d(out_ch, out_ch),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 25, out_channels: int = 8, width: int = 48):
        super().__init__()
        w = width
        self.enc1 = ConvBlock(in_channels, w)
        self.enc2 = ConvBlock(w, 2 * w)
        self.enc3 = ConvBlock(2 * w, 4 * w)
        self.pool = nn.AvgPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = ConvBlock(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = ConvBlock(2 * w, w)
        self.head = nn.Conv2d(w, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

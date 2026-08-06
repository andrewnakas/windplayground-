"""Rasp & Thuerey (2021) Resnet -- a faithful copy, not an adaptation.

The model behind the 5.625-degree WeatherBench record we spent this project
chasing: z500 RMSE 268 m^2/s^2 at 3 days (CMIP6-pretrained) and 314 (ERA5 only).

Source of truth: arXiv 2008.08626v2 section 2.2, cross-checked against the
author's own implementation at github.com/raspstephan/WeatherBench
(`src/networks.py`, `build_resnet` / `convblock` / `resblock`).

    "The basic structure is a fully convolutional Resnet with 19 residual
    blocks. Each residual block consists of two convolutional blocks, defined
    as [2D convolution -> LeakyReLU -> Batch normalization -> Dropout], after
    which the inputs to the residual layer are added to the current signal.
    The 2D convolutions inside the residual blocks have 128 channels with a
    kernel size of 3. All convolutions are periodic in longitude with zero
    padding in the latitude direction. For the first layer a simple
    convolutional block with 128 channels is used with a kernel size of 7 to
    increase the field of view. LeakyReLU is used with alpha = 0.3. Weight
    decay of 1e-5 is used for all layers. Dropout is set to 0.1."

Two details are easy to get wrong and both are deliberate here:

1. **Activation precedes normalization.** The block is Conv -> LeakyReLU ->
   BatchNorm -> Dropout, not the conventional Conv -> BatchNorm -> activation.
   The reference `convblock` exposes pre/mid/post BatchNorm placement and the
   paper specifies the post variant, so this ordering is the copied one. Our
   other models (see `unet.py`) use the conventional order; do not "fix" this
   one to match them.
2. **Weight decay is true L2 on the convolutions**, applied via the optimizer
   as plain-Adam L2 (the Keras `kernel_regularizer` equivalent), *not* decoupled
   AdamW. The trainer is responsible for that; see `WeatherResNetRT.conv_params`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from windml.models.unet import CircularConv2d

LEAKY_SLOPE = 0.3


class ConvBlock(nn.Module):
    """[PeriodicConv2D -> LeakyReLU -> BatchNorm -> Dropout], the paper's unit."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        self.conv = CircularConv2d(in_ch, out_ch, kernel_size)
        self.act = nn.LeakyReLU(LEAKY_SLOPE)
        self.norm = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.act(self.conv(x))))


class ResBlock(nn.Module):
    """Two conv blocks, then add the block input back."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.a = ConvBlock(channels, channels, kernel_size, dropout)
        self.b = ConvBlock(channels, channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b(self.a(x))


class WeatherResNetRT(nn.Module):
    """The Rasp & Thuerey Resnet.

    Defaults reproduce the paper exactly: 117 input channels (114 dynamic --
    5 variables x 7 levels + t2m + precip + TISR, at t, t-6h and t-12h -- plus
    land-sea mask, orography and latitude), 3 outputs (z500, t850, t2m), 19
    residual blocks of 128 channels, and a kernel-7 stem.

    At those defaults the parameter count is ~6.36M, against the ~6.3M the paper
    reports. `tests/test_rt_resnet.py` asserts that, because the count is the
    cheapest available check that the copy is structurally faithful -- a wrong
    block count, width or kernel size all move it.
    """

    def __init__(
        self,
        in_channels: int = 117,
        out_channels: int = 3,
        width: int = 128,
        n_blocks: int = 19,
        dropout: float = 0.1,
        stem_kernel: int = 7,
        long_skip: bool = False,
    ):
        super().__init__()
        self.stem = ConvBlock(in_channels, width, stem_kernel, dropout)
        self.blocks = nn.ModuleList(
            ResBlock(width, 3, dropout) for _ in range(n_blocks)
        )
        # No activation or normalization on the head: it regresses standardized
        # residuals, which are signed and roughly unit-scale.
        self.head = CircularConv2d(width, out_channels, 3)
        self.long_skip = long_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = stem_out = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.long_skip:
            x = x + stem_out
        return self.head(x)

    def conv_params(self) -> list[nn.Parameter]:
        """Convolution weights -- the tensors the paper's L2 1e-5 applies to.

        BatchNorm scales/offsets and all biases are excluded, matching Keras
        `kernel_regularizer`, which regularizes kernels only.
        """
        return [m.conv.weight for m in self.modules()
                if isinstance(m, CircularConv2d)]
